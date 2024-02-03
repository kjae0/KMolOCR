import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
import time
import sys
import os

from models import encoder
from models import decoder

import utils

class Caption(nn.Module):
    def __init__(self, config):
        super(Caption, self).__init__()
        self.encoder = encoder.EfficientNetV2Large()
        self.decoder = decoder.CaptionDecoder(config)
        
    def forward(self, x, tgt, tgt_padding_mask=None, tgt_mask=None):
        img_feature = self.encoder(x)
        
        if tgt_mask == None:
            tgt_mask = utils.set_up_causal_mask(tgt.shape[-1]).to(dtype=x.dtype, device=x.device)
        out = self.decoder(tgt, img_feature, tgt_padding_mask, tgt_mask)
        
        return out
