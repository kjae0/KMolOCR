from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torchvision.models._utils import IntermediateLayerGetter

import torch
import torch.nn as nn


class EfficientNetV2Large(nn.Module):
    def __init__(self, half=False):
        super(EfficientNetV2Large, self).__init__()
        self.model = efficientnet_v2_l(weight=EfficientNet_V2_L_Weights.DEFAULT)
        
        if half:
            self.model = self.model.half()
        
        # output shape -> [B, 1280, 15, 15]
        self.feature_extractor = IntermediateLayerGetter(self.model, return_layers={"features": "feature"})
        self.output_channel = 1280
        self.output_size = 15 * 15
        
    def forward(self, x):
        """
        input: B x 3 x input size x input size
        output: B x 1280 x 225
        """
        B = x.shape[0]
        
        out = self.feature_extractor(x)['feature']
        out = out.view(B, self.output_channel, self.output_size)
        
        return out

