import torch
from torch import nn
import torch.nn.functional as F

from model.utils import NestedTensor, nested_tensor_from_tensor_list 
from model.encoder import create_encoder
from model.decoder import Transformer


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x 


class Caption(nn.Module):
    def __init__(self, max_len, hidden_dimensions=256, vocab_size=70, input_dim=512, num_layers=6, encoder_model='resnet101'):
        super().__init__()
        self.encoder = create_encoder(hidden_dimensions=hidden_dimensions, encoder_model=encoder_model)
        self.input_proj = nn.Conv2d(self.encoder.num_channels, hidden_dimensions, kernel_size=1)
        self.decoder = Transformer(max_position_embeddings=max_len)
        self.ffn = FFN(hidden_dimensions, input_dim, vocab_size, num_layers)
        
    def forward(self, inputs, target, target_mask):
        if not isinstance(inputs, NestedTensor):
            inputs = nested_tensor_from_tensor_list(inputs).to(inputs.device)
            
        features, pos = self.encoder(inputs)

        src, mask = features[-1].decompose()
        hs = self.decoder(self.input_proj(src), mask, pos[-1], target, target_mask).to(inputs.device)
        out = self.ffn(hs.permute(1, 0, 2)).to(inputs.device)

        return out
    