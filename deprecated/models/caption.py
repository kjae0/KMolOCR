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
        if config['model_configuration']['encoder'] == 'v2_large':
            self.encoder = encoder.EfficientNetV2Large()
        elif config['model_configuration']['encoder'] == 'b7':
            self.encoder = encoder.EfficientNetB7()
        
        if config['model_configuration']['transformer_decoder_only']:
            self.decoder = decoder.CaptionDecoder(config)
        else:
            self.decoder = decoder.Transformer(config)
            
        self.d_model = config["model_configuration"]['d_model']
        
    def _set_new_classifier(self, num_layers, hidden_dim, output_dim):
        n_dims = [self.d_model] + [hidden_dim for _ in range(num_layers-1)] + [output_dim]
        
        new_classifier = []
        for i in range(num_layers):
            new_classifier.append(nn.Linear(n_dims[i], n_dims[i+1]))
            new_classifier.append(nn.ReLU())
        new_classifier = nn.Sequential(*new_classifier[:-1])
        
        print(new_classifier)
        self.decoder.classifier = new_classifier
        
        
    def forward(self, x, tgt, tgt_padding_mask=None, tgt_mask=None):
        img_feature = self.encoder(x)
        
        if tgt_mask == None:
            tgt_mask = utils.set_up_causal_mask(tgt.shape[-1]).to(dtype=x.dtype, device=x.device)
            
        src_mask = None
        src_padding_mask = None
        
        out = self.decoder(tgt, img_feature, 
                           src_mask, 
                           tgt_mask, 
                           src_padding_mask, 
                           tgt_padding_mask)
        
        return out
