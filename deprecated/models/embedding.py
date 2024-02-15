import torch
import torch.nn as nn

class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x


import torch
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
    return pos * angle_rates

def positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0  # d_model must be even number.
    row_pos = torch.repeat_interleave(torch.arange(row).unsqueeze(1), col, dim=0)
    col_pos = torch.repeat_interleave(torch.arange(col).unsqueeze(0), row, dim=0).reshape(-1, 1)

    angle_rads_row = get_angles(
        row_pos, torch.arange(d_model // 2).unsqueeze(0), d_model // 2
    )
    angle_rads_col = get_angles(
        col_pos, torch.arange(d_model // 2).unsqueeze(0), d_model // 2
    )

    angle_rads_row[:, 0::2] = torch.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = torch.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = torch.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = torch.cos(angle_rads_col[:, 1::2])

    pos_encoding = torch.cat([angle_rads_row, angle_rads_col], axis=1)[None, ...]
    return pos_encoding

