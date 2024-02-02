import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
import time
import sys
import os

import encoder
import decoder


class Caption(nn.Module):
    def __init__(self, config):
        super(Caption, self).__init__()
        self.is_half = config['half']
        self.encoder = encoder.EfficientNetV2Large(half=self.is_half)
        self.decoder = decoder.CaptionDecoder(config)
        
        if self.is_half:
            self.decoder = self.decoder.half()
        
    def forward(self, x, tgt, tgt_padding_mask=None, tgt_mask=None):
        img_feature = self.encoder(x)
        out = self.decoder(tgt, img_feature, tgt_padding_mask, tgt_mask)
        
        return out
