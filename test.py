from tqdm import tqdm
from torch import optim
from torch.cuda.amp import GradScaler

import torch
import torch.nn as nn
import wandb
import os

import utils
import test


def evaluate(cfg,
          model,
          criterion,
          dataloader,
          pad_idx):
    
    # torch.autograd.set_detect_anomaly(True)
    # scaler = GradScaler()
    device = cfg['device']

    loss_per_epoch = 0
    model.eval() 
    
    crt_char, wrg_char, crt_smiles, wrg_smiles = 0, 0, 0, 0

    with torch.no_grad():
        for batch_id, (img, tgt, tgt_padding_mask) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"evaluating...", ncols=100):            
            img = img.to(device)
            tgt = tgt.long().to(device)
            input_tgt = tgt[:, :-1].to(device)
            gt_tgt = tgt[:, 1:].to(device)
            tgt_padding_mask = tgt_padding_mask[:, :-1].bool().to(device)
            # tgt_mask = tgt_mask.to(device)
            tgt_mask = None
            
            # print(img.shape, input_tgt.shape, tgt_padding_mask.shape, tgt_mask.shape)
            # output -> B x max_len x vocab_size
            
            if cfg['mixed_precision']:
                with torch.autocast(device):
                    output = model(img, input_tgt, 
                                tgt_padding_mask, tgt_mask)
            else:
                output = model(img, input_tgt, 
                            tgt_padding_mask, tgt_mask)
            
            # output -> (B * max_len) x vocab_size
            output_shape = output.shape     # (B, max_len, vocab_size)
            output = output.contiguous().view(-1, output.shape[-1])
            gt_tgt = gt_tgt.contiguous().view(-1)
            
            loss = criterion(output, gt_tgt)
            loss_per_epoch += loss.item()
            
            prediction = torch.argmax(output, dim=1)
            prediction = prediction.view(output_shape[0], -1)
            ground_truth = gt_tgt.view(output_shape[0], -1)
            
            cc, wc = utils.eval_char_unit(prediction, ground_truth, pad_idx)
            cs, ws = utils.eval_smiles_unit(prediction, ground_truth)
            
            crt_char += cc.item()
            wrg_char += wc.item()
            crt_smiles += cs.item()
            wrg_smiles += ws.item()
            
        print("character unit accuracy: ", crt_char, wrg_char, crt_char/(crt_char+wrg_char))
        print("smiles unit accuracy: ", crt_smiles, wrg_smiles, crt_smiles/(crt_smiles+wrg_smiles))
        
        wandb.log({"character unit accuracy": crt_char/(crt_char+wrg_char)})
        wandb.log({"smiles unit accuracy": crt_smiles/(crt_smiles+wrg_smiles)})
            
        print("prediction")
        print(prediction[:5], sep="\n")
        print()
        
        print("gt_tgt")
        print(ground_truth[:5], sep="\n")
        print()
            
        print("val loss:", loss_per_epoch / len(dataloader))
        wandb.log({'val loss':(loss_per_epoch / len(dataloader))})

# test eval
# wandb
        