from tqdm import tqdm

import torch
import torch.nn as nn
import utils

from transformer.models.embedding import transformer_embedding, token_embeddings, positional_encoding

def make_pad_mask(q, k, q_pad_idx, k_pad_idx):
    len_q, len_k = q.size(1), k.size(1)

    # batch_size x 1 x 1 x len_k
    k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
    # batch_size x 1 x len_q x len_k
    k = k.repeat(1, 1, len_q, 1)

    # batch_size x 1 x len_q x 1
    q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
    # batch_size x 1 x len_q x len_k
    q = q.repeat(1, 1, 1, len_k)

    mask = k & q
    return mask

def make_no_peak_mask(q, k, device='cuda'):
    len_q, len_k = q.size(1), k.size(1)

    # len_q x len_k
    mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(device)

    return mask

def inference(cfg,
            backbone,
            seq2seq,
            dataloader,
            criterion
            ):
    
    device = cfg.device
    
    seq2seq.eval()
    backbone.eval()
    
    pos_emb = positional_encoding.PostionalEncoding(d_model=cfg.d_model, max_len=100, device='cpu')
    emb = transformer_embedding.TransformerEmbedding(vocab_size=cfg.vocab_size,
                                                     d_model=cfg.d_model,
                                                     max_len=100,
                                                     drop_prob=0,
                                                     device='cpu',
                                                     token_embed=True)
    losses = []
    pos_enc = True
    
    with torch.no_grad():
        predictions = []
        labels = []
        loss_per_epoch = 0
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f"inference...", ncols=60):
            x = x.to(device)
            y = y.long()
            input_tgt = y[:, :-1]
            input_tgt_embedded = emb(input_tgt).to(device)
            label_tgt = y[:, 1:].to(device)
            input_tgt = input_tgt.to(device)
            # print(x.device, input_tgt.device, input_tgt_embedded.device)
            
            encoded = backbone(x)
            if pos_enc:
                x_pos_enc = pos_emb(encoded.cpu()).to(encoded.device)
                pos_enc = False
                
            encoded += x_pos_enc
            
            trg_mask = make_pad_mask(input_tgt, input_tgt, cfg.pad_idx, cfg.pad_idx) * \
                        make_no_peak_mask(input_tgt, input_tgt).to(device)
                        
            output = seq2seq(src=encoded, 
                            #  trg=input_tgt, 
                             trg_embedded=input_tgt_embedded,
                             trg_mask=trg_mask)
            
            predictions.append(output.detach().cpu())
            labels.append(label_tgt.cpu())
            
            output = output.contiguous().view(-1, output.shape[-1])
            label_tgt = label_tgt.contiguous().view(-1)
            
            loss = criterion(output, label_tgt)
            loss_per_epoch += loss.item()
            
            
        print(f"test loss : {loss_per_epoch/len(dataloader)}\n")    
        
        # predictions = torch.concat(predictions, dim=0)
        # labels = torch.concat(labels, dim=0)
        predictions = utils.retrieve_from_output(predictions, cfg.i2c, cfg.eos_idx)
        labels = utils.retrieve_from_logit(labels, cfg.i2c, cfg.eos_idx)
        print(utils.accracy_score(predictions, labels))
                      
    # return seq2seq        
