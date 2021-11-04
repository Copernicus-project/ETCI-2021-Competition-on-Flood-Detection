from glob import glob
import os
import cv2
import numpy as np
import pandas as pd
import imageio
from torch.utils.data import DataLoader

from src import config
from src.etci_dataset import ETCIDataset
from src.train import get_dataloader_un,get_dataloader
from src.utils import dataset_utils, worker_utils, sampler_utils


train_loader, val_loader = get_dataloader(0, 1)
print(train_loader)

for i in train_loader:
    print(i)

path = 'Y:/Users/Shuo/tiles_big_dataset_label_undersample_v2/'

train_df = dataset_utils.create_df(config.train_dir)
train_dataset = ETCIDataset(train_df, split="train")

# determine if an image has mask or not
flood_label_paths = train_df["flood_label_path"].values.tolist()
train_has_masks = list(map(dataset_utils.has_mask, flood_label_paths))
train_df["has_mask"] = train_has_masks

# create samplers
stratified_sampler = sampler_utils.BalanceClassSampler(
    train_df["has_mask"].values.astype("int")
)
train_sampler = sampler_utils.DistributedSamplerWrapper(
    stratified_sampler, rank=0, num_replicas=1, shuffle=True
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

for batch in train_loader:
    print(batch)
