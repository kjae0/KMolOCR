from tqdm import tqdm

import numpy as np
import random
import torch
import os

def set_up_causal_mask(seq_len):
    """Defines the triangular mask used in transformers.

    This mask prevents decoder from attending the tokens after the current one.

    Arguments:
        seq_len (int): Maximum length of input sequence
    Returns:
        mask (torch.Tensor): Created triangular mask
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
    mask.requires_grad = False
    return mask

def smiles_processer(smiles, max_len, sos_idx, eos_idx, pad_idx, vocab):
    ret = [sos_idx]
        
    for s in smiles:
        if s in vocab:
            ret.append(vocab[s])
        else:
            ret.append(vocab['<unk>'])

    ret.append(eos_idx)
    ret.extend([pad_idx for _ in range(max_len-len(ret))])
    return ret

def save_ckpt(ckpt, save_dir, epoch, unit):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    torch.save(ckpt, os.path.join(save_dir, f"{epoch}_{unit}.pt"))
    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 속도 저하를 방지하기 위해 제외
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def eval_char_unit(prediction, ground_truth, pad_idx):
    crt, wrg = 0, 0
    
    match_mask = (prediction == ground_truth).long() * 10
    pad_mask = (ground_truth != pad_idx).long()
    
    match_mask += pad_mask
    crt += (match_mask == 11).sum()
    wrg += (match_mask == 1).sum()
    
    return crt, wrg
    
def eval_smiles_unit(prediction, ground_truth):
    crt, wrg = 0, 0
    
    match_mask = (prediction == ground_truth)
    match_mask = match_mask.all(dim=1)
    
    crt += match_mask.sum()
    wrg += len(match_mask) - match_mask.sum()
    
    return crt, wrg
            
