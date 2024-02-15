import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class ResidualBlock(nn.Module):
    """Represents 1D version of the residual block: https://arxiv.org/abs/1512.03385"""

    def __init__(self, input_dim):
        """Initializes the module."""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        """Performs forward pass of the module."""
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x
        

class PositionalEncoding(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncoding, self).__init__()
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


class CaptionDecoder(nn.Module):
    """Decoder for image captions.

    Generates prediction for next caption word given the prviously
    generated word and image features extracted from CNN.    
    """

    def __init__(self, config):
        """Initializes the model."""
        super(CaptionDecoder, self).__init__()
        model_config = config["model_configuration"]
        decoder_layers = model_config["decoder_layers"]
        attention_heads = model_config["attention_heads"]
        d_model = model_config["d_model"]
        ff_dim = model_config["ff_dim"]
        dropout = model_config["dropout"]
        vocab_size = config["vocab_size"]
        img_feature_channels = config["image_specs"]["img_feature_channels"]
        
        self.embedding_layer = nn.Embedding(config['vocab_size'],
                                            config['embedding']['embedding_dim'],
                                            config['PAD_idx'])

        self.entry_mapping_words = nn.Linear(config['embedding']['embedding_dim'], d_model)
        self.entry_mapping_img = nn.Linear(img_feature_channels, d_model)

        self.res_block = ResidualBlock(d_model)

        self.positional_encodings = PositionalEncoding(config["max_len"]-1, d_model, dropout)
        
        
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, image_features, 
                src_mask=None,
                tgt_mask=None,
                src_padding_mask=None,
                tgt_padding_mask=None
                ):
        """Performs forward pass of the module."""
        # Adapt the dimensionality of the features for image patches
        
        # B x L x D -> B x L x d_model
        image_features = self.entry_mapping_img(image_features)
        
        # B x L x d_model -> L x B x d_model
        image_features = image_features.permute(1, 0, 2)
        image_features = F.leaky_relu(image_features)

        # Entry mapping for word tokens
        x = self.embedding_layer(x)
        x = self.entry_mapping_words(x)
        x = F.leaky_relu(x)

        x = self.res_block(x)
        x = F.leaky_relu(x)

        x = self.positional_encodings(x)

        # Get output from the decoder
        x = x.permute(1, 0, 2)
        x = self.decoder(
            tgt=x,
            memory=image_features,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        x = x.permute(1, 0, 2)

        x = self.classifier(x)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.embed_size = cfg['embedding']['embedding_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg["model_configuration"]['dropout']
        self.d_model = cfg["model_configuration"]['d_model']
        self.dim_feedforward = cfg["model_configuration"]["ff_dim"]
        self.num_encoder_layers = cfg["model_configuration"]['encoder_layers']
        self.num_decoder_layers = cfg["model_configuration"]['decoder_layers']
        self.nheads = cfg["model_configuration"]['attention_heads']
        self.img_feature_channels = cfg["image_specs"]["img_feature_channels"]
        

        self.tgt_embed = nn.Embedding(self.vocab_size, 
                                      self.embed_size,
                                      cfg['PAD_idx'])
        self.positional_encoding = PositionalEncoding(cfg["max_len"]-1, self.d_model, self.dropout)

        self.mapping_words = nn.Linear(self.embed_size, self.d_model)
        self.mapping_img = nn.Linear(self.img_feature_channels, self.d_model)

        self.transformer = nn.Transformer(d_model=self.d_model, 
                                          nhead=self.nheads, 
                                          num_encoder_layers=self.num_encoder_layers, 
                                          num_decoder_layers=self.num_decoder_layers, 
                                          dim_feedforward=self.dim_feedforward, 
                                          dropout=self.dropout)

        # Output linear layer
        self.classifier = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, tgt, src,
                src_mask=None, 
                tgt_mask=None, 
                src_padding_mask=None, 
                tgt_padding_mask=None):
        # src: feature vectors from CNN encoder, shape: [src_len, batch_size, embed_size]
        # tgt: input tokens for caption, shape: [tgt_len, batch_size]
        
        # B x L x feature_dim -> B x L x d_model
        src = self.mapping_img(src)
        
        
        tgt = self.tgt_embed(tgt) * math.sqrt(self.embed_size)
        # B x L x embed_dim -> B x L x d_model
        tgt = self.mapping_words(tgt)
        tgt = self.positional_encoding(tgt)

        # permute or set batch_first=True
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        
        # Transformer forward pass
        output = self.transformer(src, tgt, 
                                  src_mask=src_mask, 
                                  tgt_mask=tgt_mask, 
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        
        output = output.permute(1, 0, 2)
        
        # Pass through the output layer
        return self.classifier(output)
    
