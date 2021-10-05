import os
import segmentation_models_pytorch as smp

# dataset related
dataset_root = ""
train_dir = os.path.join(dataset_root, "../train/train/")
valid_dir = os.path.join(dataset_root, "../train/train/valid/")
test_dir = os.path.join(dataset_root, "../train/train/valid/")
local_batch_size = 2

# model related
backbone = "mobilenet_v2"
model_family = smp.Unet

# training related
learning_rate = 1e-3
model_serialization = "unet_mobilenet_v2"
num_epochs = 15
