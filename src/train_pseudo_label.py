"""
Ideas implemented in this script comes from here:
https://www.youtube.com/watch?v=SsnWM1xWDu4. This script should be run
after executing the `notebook/Generate_Pseudo.ipynb` notebook. Also,
take note of the `pseudo_df` variable in the `get_dataloader()` function.
"""

import argparse
import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp
import torch

from etci_dataset import ETCIDataset, ETCIDatasetUN
from utils import sampler_utils
from utils import dataset_utils
from utils import metric_utils
from utils import worker_utils
import config

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# fix all the seeds and disable non-deterministic CUDA backends for
# reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# set up logging
import logging

logging.basicConfig(
    filename="pseudo_label.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def get_dataloader(rank, world_size):
    """Creates the data loaders."""
    # create dataframes
    train_df = dataset_utils.create_df_un(config.train_dir)
    valid_df = dataset_utils.create_df_un(config.valid_dir)
    # this path depends on where you have serialized the dataframe while
    # executing `notebook/Generate_Pseudo.ipynb`.

    pseudo_df_path = "../notebooks/pseudo_df.csv"
    pseudo_df = pd.read_csv(pseudo_df_path)
    length = int(len(pseudo_df) / 5)
    train_df = pd.concat([train_df[0:length], pseudo_df[0:length]], axis=1)
    # determine if an image has mask or not
    flood_label_paths = train_df["flood_label_path2"].values.tolist()
    train_has_masks = list(map(dataset_utils.has_mask, flood_label_paths))
    train_df["has_mask"] = train_has_masks

    # filter invalid images
    remove_indices = dataset_utils.filter_df_un(train_df)
    train_df = train_df.drop(train_df.index[remove_indices])

    # define augmentation transforms
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(270),
            A.ElasticTransform(
                p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            ),
        ]
    )

    # define dataset
    train_dataset = ETCIDatasetUN(train_df, split="train", transform=transform)
    validation_dataset = ETCIDatasetUN(valid_df[0:length], split="validation", transform=None)

    # create samplers
    stratified_sampler = sampler_utils.BalanceClassSampler(
        train_df["has_mask"].values.astype("int")
    )
    train_sampler = sampler_utils.DistributedSamplerWrapper(
        stratified_sampler, rank=rank, num_replicas=world_size, shuffle=True
    )
    val_sampler = DistributedSampler(
        validation_dataset, rank=rank, num_replicas=world_size, shuffle=False
    )

    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=1,
        worker_init_fn=worker_utils.seed_worker,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=1,
    )

    return train_loader, val_loader


def create_model(weight_path):
    """Initializes a segmentation model and loads the weights into it.

    Args:
        weight_path: Path to the pre-trained model weights.
    """
    model = smp.Unet(
        encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=2
    )
    weights = torch.load(weight_path)
    model.load_state_dict(weights)
    return model


def train(rank, num_epochs, world_size, pretrained_path, finetune_path):
    """Fine-tunes the segmentation model using distributed training."""
    # initialize the workers and fix the seeds
    worker_utils.init_process(rank, world_size)
    torch.manual_seed(0)

    # model loading and off-loading to the current device
    model = create_model(pretrained_path)
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    # configure optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # define loss function and gradient scaler
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # get data loaders
    train_loader, val_loader = get_dataloader(rank, int(world_size))
    print(train_loader)
    ## begin training ##
    for epoch in range(num_epochs):
        print(epoch)
        losses = metric_utils.AvgMeter()
        if rank == 0:
            logging.info(
                "Rank: {}/{} Epoch: [{}/{}]".format(
                    rank, world_size, epoch + 1, num_epochs
                )
            )

        # train set
        model.train()
        for batch in train_loader:
            with torch.cuda.amp.autocast(enabled=True):
                image = batch["image"].cuda(rank, non_blocking=True)
                mask = batch["mask"].cuda(rank, non_blocking=True)
                pred = model(image)

                loss = criterion_dice(pred, mask)
                losses.update(loss.cpu().item(), image.size(0))

            # update the model
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        loss = losses.avg
        global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)

        if rank == 0:
            logging.info(f"Epoch: {epoch+1} learning rate: {scheduler.get_last_lr()}")
            logging.info(f"Epoch: {epoch+1} Train Loss: {global_loss[0]:.3f}")

        ## evaluation ##
        if epoch % 5 == 0:
            if rank == 0:
                logging.info("Running evaluation on the validation set.")
            model.eval()
            losses = metric_utils.AvgMeter()

            with torch.no_grad():
                for batch in val_loader:
                    with torch.cuda.amp.autocast(enabled=True):
                        image = batch["image"].cuda(rank, non_blocking=True)
                        mask = batch["mask"].cuda(rank, non_blocking=True)
                        pred = model(image)

                        loss = criterion_dice(pred, mask)
                        losses.update(loss.cpu().item(), image.size(0))

            loss = losses.avg
            global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)
            print(global_loss)
            if rank == 0:
                logging.info(f"Epoch: {epoch + 1} Val Loss: {global_loss[0]:.3f}")

    if rank == 0:
        torch.save(model.module.state_dict(), f"{finetune_path}_{rank}.pth")


WORLD_SIZE = torch.cuda.device_count()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, default=20, help="number of epochs")
    ap.add_argument(
        "-p",
        "--pretrained_path",
        type=str,
        help="paths to the pretrained weights (with .pth)",
        required=True,
        default='./pretrained_unet_mobilenet_v2_0.pth'
    )
    ap.add_argument(
        "-f",
        "--finetune-path",
        type=str,
        help="paths to the weights to be serialized after fine-tuning (without .pth)",
        required=True,
        default='../new_weights/new_unet_v3.pth'
    )
    args = vars(ap.parse_args())
    args["pretrained_path"] = './pretrained_unet_mobilenet_v2_0.pth'
    print(WORLD_SIZE)
    train(0, args["epochs"], WORLD_SIZE, args["pretrained_path"], args["finetune_path"])
    # mp.spawn(
    #     train,
    #     args=(
    #         args["epochs"],
    #         WORLD_SIZE,
    #         args["pretrained_path"],
    #         args["finetune_path"],
    #     ),
    #     nprocs=WORLD_SIZE,
    #     join=True,
    # )
