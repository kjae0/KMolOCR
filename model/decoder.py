import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, fc_dim=2048, dropout=0.1,
                 act_fn=nn.ReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(fc_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act_fn = act_fn
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act_fn(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.act_fn(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, fc_dim=2048, dropout=0.1,
                 act_fn=nn.ReLU, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(fc_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.act_fn = act_fn
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.act_fn(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.act_fn(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        tgt = tgt.to(tgt.device)
        memory = memory.to(tgt.device)
        tgt_mask = tgt_mask.to(tgt.device)
        #memory_mask = memory_mask.to(tgt.device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(tgt.device)
        memory_key_padding_mask = memory_key_padding_mask.to(tgt.device)
        pos = pos.to(tgt.device)
        query_pos = query_pos.to(tgt.device)
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size=717, hidden_dim=256, pad_token_id=277, max_position_embeddings=128, dropout=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size,hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.LayerNorm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(x.device)
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        input_embeds = self.word_embeddings(x).to(device)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, fc_dim, dropout, num_layers, act_fn, norm=True):
        super().__init__()
        self.layers = [TransformerEncoderLayer(d_model, n_head, fc_dim,dropout, act_fn, norm) for _ in range(num_layers)]
        self.num_layers = num_layers
        
        if norm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None

    def forward(self, src,
                mask = None,
                src_key_padding_mask = None,
                pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, fc_dim, act_fn, num_layers, dropout, norm=True, return_intermediate=False):
        super().__init__()
        self.layers = [TransformerDecoderLayer(d_model, n_head, fc_dim, dropout, act_fn, norm) for _ in range(num_layers)]
        
        if norm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
    
    

class Transformer(nn.Module):
    def __init__(self, vocab_size=717, hidden_dim=256, pad_token_id=0, 
                 max_position_embeddings=100, dropout=0.1, layer_norm_eps=1e-12,
                 d_model=256, n_head=8, num_encoder_layers=3,
                 num_decoder_layers=3, fc_dim=2048,
                 act_fn=nn.ReLU, normalize_before=True,
                 return_intermediate_dec=False):
        
        super().__init__()
        self.encoder = TransformerEncoder(d_model=d_model,
                                          n_head=n_head,
                                          fc_dim=fc_dim,
                                          dropout=dropout,
                                          num_layers=num_encoder_layers,
                                          act_fn=act_fn,
                                          norm=normalize_before)        
        self.decoder = TransformerDecoder(d_model=d_model,
                                            n_head=n_head,
                                            fc_dim=fc_dim,
                                            act_fn=act_fn,
                                            num_layers=num_decoder_layers,
                                            dropout=dropout,
                                            norm=normalize_before,
                                            return_intermediate=return_intermediate_dec)
        self.embeddings = DecoderEmbeddings(vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout, layer_norm_eps)
        
        self._reset_parameters()
        self.d_model = d_model
        self.n_head = n_head

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, tgt, tgt_mask):
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        mask = mask.flatten(1)
        
        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, B, 1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
        return hs

