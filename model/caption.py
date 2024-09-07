import torch
from torch import nn
import torch.nn.functional as F

from model.utils import NestedTensor, nested_tensor_from_tensor_list 
from model.encoder import create_encoder
from model.decoder import Transformer
from model.ffn import FFN
from model.pe import PositionEmbeddingSine, PositionEmbeddingLearned


class Caption(nn.Module):
    def __init__(self, max_len, hidden_dimensions=256, vocab_size=70, input_dim=512, num_layers=6, encoder_model='resnet101'):
        super().__init__()
        self.encoder = create_encoder(hidden_dimensions=hidden_dimensions, encoder_model=encoder_model)
        self.pe = PositionEmbeddingLearned(num_pos_feats=hidden_dimensions)
        self.input_proj = nn.Conv2d(self.encoder.num_channels, hidden_dimensions, kernel_size=1)
        self.decoder = Transformer(max_position_embeddings=max_len)
        self.ffn = FFN(hidden_dimensions, input_dim, vocab_size, num_layers)
        
    def forward(self, inputs, target, target_mask):
        features= self.encoder(inputs)
        pos = self.pe(features[-1])

        src = features[-1]
        b, c, h, w = src.shape
        mask = torch.zeros((b, h, w)).bool().to(inputs.device)
        
        hs = self.decoder(self.input_proj(src), mask, pos[-1], target, target_mask).to(inputs.device)
        out = self.ffn(hs.permute(1, 0, 2)).to(inputs.device)

        return out
    