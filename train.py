import torch
import torch.nn as nn
import wandb
import numpy as np
import random

# from ray import train as train_ray
from tqdm import tqdm

import test
import utils

def train_one_epoch(model, criterion, optimizer, scheduler, train_dl, cfg, epoch, device):
    model.train()
    criterion.train()
    loss_per_epoch = 0
    
    for batch_id, (x, y, _) in tqdm(enumerate(train_dl), desc=f"{epoch+1} epoch training...", ncols=60, total=len(train_dl)):
        # break
        x = x.to(device)
        y = y.to(device)

        input_tgt = y[:, :-1].to(device)
        label_tgt = y[:, 1:].to(device)
        target_mask = utils.make_pad_mask(input_tgt, cfg.pad_idx)
        
        out = model(x, input_tgt, target_mask)
        out = out.contiguous().view(-1, out.shape[-1])
        label_tgt = label_tgt.contiguous().view(-1)

        loss = criterion(out, label_tgt)
        loss_per_epoch += loss.item()
        
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.zero_grad()
        else:
            optimizer.zero_grad()
        
        loss.backward()
        
        if isinstance(optimizer, list):
            optimizer[0].step()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), cfg['clip_norm'])
            optimizer[1].step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_norm'])
            optimizer.step()
            
        scheduler.step()
        
        if (batch_id + 1) % (cfg.print_interval) == 0:
            print()
            print(batch_id, loss.item())
            wandb.log({'train loss': loss.item()})
            
        if (batch_id + 1) % (cfg.save_interval) == 0:
            print(f"{epoch}_{batch_id} checkpoint saved.")
            utils.save_ckpt({'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'seed': {
                                    'pytorch': torch.get_rng_state(),
                                    'pytorch_cuda': torch.cuda.get_rng_state(),
                                    'numpy': np.random.get_state(),
                                    'random': random.getstate()
                                }
                                },
                            cfg.save_ckpt_dir, 
                            f"{epoch}_{batch_id}_ckpt.pt")
                    
    return loss_per_epoch / len(train_dl)

def evaluate(cfg, model, test_dl, criterion):
    device = cfg.device
    model.eval()
    criterion.eval()
    
    losses = 0
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
            target_mask = utils.make_pad_mask(input_tgt, cfg.pad_idx)
            
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
            
def train(cfg, model, train_dl, test_dl, criterion, optimizer, scheduler):
    device = cfg.device

    for epoch in range(cfg.num_epochs):
        train_one_epoch(model, criterion, optimizer, train_dl, cfg, epoch, device)
        
        if isinstance(scheduler, list):
            for sche in scheduler:
                sche.step()
        else:
            scheduler.step()
            
        if (epoch + 1) % (cfg.eval_interval) == 0:
            smiles_acc, chr_acc, test_loss = evaluate(cfg, model, test_dl, criterion)
            
            print(f"test loss: {test_loss}")
            print(f"test smiles acc: {smiles_acc}")
            print(f"test chr acc: {chr_acc}")
            
            wandb.log({'test loss': test_loss})
            wandb.log({'test smiles acc': smiles_acc})
            wandb.log({'test chr acc': chr_acc})
