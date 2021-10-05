import torch
import segmentation_models_pytorch as smp

model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=2)
state_dict = torch.load('unet_pseudo_mobilenetv2_round2_0.pth')
model.load_state_dict(state_dict)
print(model.eval())