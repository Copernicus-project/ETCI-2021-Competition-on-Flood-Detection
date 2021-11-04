import os
import segmentation_models_pytorch as smp

# dataset related
dataset_root = ""
# train_dir = os.path.join(dataset_root, "../train/train/")
# valid_dir = os.path.join(dataset_root, "../train/valid/")
# test_dir = os.path.join(dataset_root, "../train/valid/")
train_dir = os.path.join(dataset_root, "Y:/Users/Shuo/tiles_big_dataset_img_undersample_v2/")
valid_dir = os.path.join(dataset_root, "Y:/Users/Shuo/tiles_big_dataset_img_undersample_v2/")
test_dir = os.path.join(dataset_root, "Y:/Users/Shuo/tiles_big_dataset_img_undersample_v2/")
local_batch_size = 1

# model related
backbone = "mobilenet_v2"
model_family = smp.Unet

# training related
learning_rate = 1e-3
model_serialization = "pretrained_unet_mobilenet_v2"
num_epochs = 16
