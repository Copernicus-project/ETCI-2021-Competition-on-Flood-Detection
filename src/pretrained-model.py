import torch
from torch.nn.parallel import DistributedDataParallel

import config
import numpy as np
from utils import metric_utils
from utils import worker_utils
from train import get_dataloader_un, create_model
# set up logging
import logging
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp
import warnings

warnings.filterwarnings("ignore")

# fix all the seeds and disable non-deterministic CUDA backends for
# reproducibility
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(0)

logging.basicConfig(
    filename="pretrained_training.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

def trainun(rank, num_epochs, world_size):
    """Trains the segmentation model using distributed training."""
    # initialize the workers and fix the seeds
    worker_utils.init_process(rank, world_size)
    torch.manual_seed(0)

    print("pretrained eval")
    model = create_model(smp.Unet, "mobilenet_v2")
    state_dict = torch.load('../weights/unet_pseudo_mobilenetv2_round2_0.pth')
    model.load_state_dict(state_dict)
    torch.cuda.set_device(rank)
    model.cuda(rank)
    #
    # torch.cuda.set_device(rank)
    # model.cuda(rank)
    # model = DistributedDataParallel(model, device_ids=[rank])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # set up loss function and gradient scaler for mixed-precision
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # initialize data loaders
    train_loader, val_loader = get_dataloader_un(rank, world_size)

    # begin training
    for epoch in range(num_epochs):
        print("epoch:", epoch)
        losses = metric_utils.AvgMeter()

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

        # average out the losses and log the summary
        loss = losses.avg
        global_loss = metric_utils.global_meters_all_avg(rank, 1, loss)

        if rank == 0:
            logging.info(f"Epoch: {epoch + 1} Train Loss: {global_loss[0]:.3f}")

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
            print(loss)
            global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)
            if rank == 0:
                logging.info(f"Epoch: {epoch + 1} Val Loss: {global_loss[0]:.3f}")
            print("losses:", losses)
    # serialization of model weights
    # if rank == 0:
    #     torch.save(
    #         model.state_dict(), f"{config.model_serialization}_{rank}.pth"
    #     )

WORLD_SIZE = torch.cuda.device_count()

def evaluate_model(rank, num_epochs, world_size):
    print("pretrained eval")
    model = create_model(smp.Unet, "mobilenet_v2")
    state_dict = torch.load('../weights/unet_pseudo_mobilenetv2_round2_0.pth')
    model.load_state_dict(state_dict)
    torch.cuda.set_device(rank)
    model.cuda(rank)
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")
    train_loader, val_loader = get_dataloader_un(rank, world_size)
    losses = metric_utils.AvgMeter()
    model.eval()
    for batch in train_loader:
        with torch.cuda.amp.autocast(enabled=True):
            image = batch["image"].cuda(rank, non_blocking=True)
            mask = batch["mask"].cuda(rank, non_blocking=True)

            pred = model(image)
            # Set total and correct
            _, predicted = torch.max(pred.data, 1)
            loss = criterion_dice(pred, mask)
            losses.update(loss.cpu().item(), image.size(0))
            print(loss)
    loss_ave = losses.avg
    # global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss_ave)
    print(loss_ave)
    # Print accuracy
    #print('Accuracy: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    print("start")
    trainun(0, config.num_epochs, WORLD_SIZE)
    #evaluate_model(0, config.num_epochs, WORLD_SIZE)
    #mp.spawn(trainun, args=(config.num_epochs, WORLD_SIZE), nprocs=WORLD_SIZE, join=True)
    print("finish")
