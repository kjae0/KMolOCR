import torch
import torch.nn as nn
import wandb
import numpy as np
import random

# from ray import train as train_ray
from tqdm import tqdm

import test
import utils

def make_pad_mask(tgt, pad_idx):
    mask = tgt == pad_idx
    return mask

def train(cfg, model, train_dl, test_dl, train_test_dl, criterion, optimizer, scheduler, postprocessor):
    device = cfg.device

    for epoch in range(cfg.num_epochs):
        model.train()
        criterion.train
        loss_per_epoch = 0
        
        for batch_id, (x, y, _) in tqdm(enumerate(train_dl), desc=f"{epoch+1} epoch training...", ncols=60, total=len(train_dl)):
            # break
            x = x.to(device)
            y = y.to(device)

            input_tgt = y[:, :-1].to(device)
            label_tgt = y[:, 1:].to(device)
            target_mask = make_pad_mask(input_tgt, cfg.pad_idx)
            
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
                # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 10)
                optimizer[1].step()
            else:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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
                
            if (batch_id + 1) % (cfg.eval_interval) == 0:
                smiles_acc, chr_acc, test_loss = test.evaluate(cfg, model, test_dl, criterion)
                
                print(f"test loss: {test_loss}")
                print(f"test smiles acc: {smiles_acc}")
                print(f"test chr acc: {chr_acc}")
                
                wandb.log({'test loss': test_loss})
                wandb.log({'test smiles acc': smiles_acc})
                wandb.log({'test chr acc': chr_acc})
            
                
        if isinstance(scheduler, list):
            for sche in scheduler:
                sche.step()
        else:
            scheduler.step()

            
            
                
                
    