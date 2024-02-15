import torch
import torch.nn as nn
import numpy as np

# from ray import train as train_ray
from tqdm import tqdm

import utils

def make_pad_mask(tgt, pad_idx):
    mask = tgt == pad_idx
    return mask

def evaluate(cfg, model, test_dl, criterion):
    device = cfg.device
    
    model.eval()
    criterion.eval()
    losses = 0
    outputs = []
    labels = []
    smiles_crt = 0
    smiles_wrg = 0
    chr_crt = 0
    chr_wrg = 0

    
    with torch.no_grad():
        for batch_id, (x, y, _) in tqdm(enumerate(test_dl), desc=f"evaluate...", ncols=60, total=len(test_dl)):
            if cfg.eval_limit and batch_id == cfg.eval_limit:
                break
            
            x = x.to(device)
            y = y.to(device)

            input_tgt = y[:, :-1].to(device)
            label_tgt = y[:, 1:].to(device)
            target_mask = make_pad_mask(input_tgt, cfg.pad_idx)
            
            out = model(x, input_tgt, target_mask)

            out = out.contiguous().view(-1, out.shape[-1])
            label_tgt = label_tgt.contiguous().view(-1)

            loss = criterion(out, label_tgt)
            losses += loss.item()
            
            # (B*100) x 70
            out = out.cpu()            
            label_tgt = label_tgt.cpu()
            
            # (B*100)
            out = torch.argmax(out, dim=1)
            
            prediction = out.view(-1, cfg.max_len)
            ground_truth = label_tgt.view(-1, cfg.max_len)
            
            cc, wc = utils.eval_char_unit(prediction, ground_truth, cfg.pad_idx)
            cs, ws = utils.eval_smiles_unit(prediction, ground_truth)

            smiles_crt += cs
            smiles_wrg += ws
            chr_crt += cc
            chr_wrg += wc
            # outputs.extend([out[idx:idx+cfg.max_len].unsqueeze(0).cpu() for idx in range(0, len(out), 100)])
            # labels.extend([label_tgt[idx:idx+cfg.max_len].unsqueeze(0).cpu() for idx in range(0, len(out), 100)])  


    losses = losses / batch_id

    return smiles_crt / (smiles_crt + smiles_wrg), \
        chr_crt / (chr_crt + chr_wrg), losses
            
            
                
                
    