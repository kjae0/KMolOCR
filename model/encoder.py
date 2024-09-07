from torchvision.models._utils import IntermediateLayerGetter
from torch import nn
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision

CHANNEL_DICT = {
    'resnet101': 2048,
    'efficientnet_b3': 1536,
    'efficientnet_b4': 1792,
    'efficientnet_b5': 2048,
    'efficientnet_b7': 2560,
    'efficientnet_v2_l': 1280,
}

def create_encoder(encoder_model):
    if encoder_model not in CHANNEL_DICT:
        raise ValueError(f"Invalid encoder model: {encoder_model}")
    
    backbone = build_backbone(encoder_model)
    model = Encoder(backbone=backbone)
    model.num_channels = CHANNEL_DICT[encoder_model]
    
    return model

def build_backbone(encoder_model):
    encoder = getattr(torchvision.models, encoder_model)(replace_stride_with_dilation=[False, False, True],pretrained=True, norm_layer=FrozenBatchNorm2d)
    return_layers = {'layer4': "0"}
    encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)
    
    return encoder, return_layers


class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class Encoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.body = backbone

    def forward(self, x):
        out = self.body(x)
        return out
    