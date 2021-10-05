import torch
from torch import nn
from torch import Tensor
from typing import Callable, Any, Optional, List

model_urls = {
    'mobilenet_v2': '../weights/unet_pseudo_mobilenetv2_round2_0.pth',
}

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
state_dict = torch.load(model_urls['mobilenet_v2'])
print(state_dict)
model.load_state_dict(state_dict)
print(model.eval())
