import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
import time
import sys
import os

from models_deprecated import encoder
from models_deprecated import decoder
from models_deprecated import pe
from models_deprecated import utils

import utils

class Caption(nn.Module):
    def __init__(self, config, state_dict_dir=None):
        super(Caption, self).__init__()
        if config['model_configuration']['encoder'] == 'v2_large':
            self.encoder = encoder.EfficientNetV2Large()
            self.img_feature_dim = 1280
        elif config['model_configuration']['encoder'] == 'b7':
            self.encoder = encoder.EfficientNetB7()
            self.img_feature_dim = 2560
        
        self.img_feature_mapping = nn.Linear(self.img_feature_dim, 256)
        self.decoder = decoder.Transformer(max_position_embeddings=config['max_len']-1)
        
        if state_dict_dir:
            print("state dict loaded!")
            state_dict = torch.load(state_dict_dir)['weight']
            self._load_decoder_state_dict(state_dict)
            
        self.positional_encoding_src = pe.PositionEmbeddingSine()
        self.classifier = nn.Linear(256, config['vocab_size'])
        
    def _load_decoder_state_dict(self, state_dict):
        filtered_state_dict = {k.replace("module.decoder", ""): v for k, v in state_dict.items() if 'decoder' in k and 'embedding' not in k}    
        self.decoder.load_state_dict(filtered_state_dict, strict=False)
        
    def forward(self, x, tgt, tgt_padding_mask=None, tgt_mask=None):
        # B x C x L
        img_feature = self.encoder(x)
        
        # B x C x dmodel
        img_feature = self.img_feature_mapping(img_feature)
        mask = torch.ones((img_feature.shape[0], self.encoder.output_size, self.encoder.output_size)).bool().to(img_feature.device)
        # pos_embed = self.positional_encoding_src(img_feature, mask)
        pos_embed = None
        mask = mask.flatten(1)
        # print("mask shape", mask.shape)
                
        out = self.decoder(img_feature, 
                           mask, # B x img_feature width x height, all one bool tensor
                           tgt,
                           tgt_padding_mask,
                           pos_embed=pos_embed)

        return self.classifier(out)
