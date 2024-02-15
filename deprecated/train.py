from tqdm import tqdm
from torch import optim
from torch.cuda.amp import GradScaler

import torch
import torch.nn as nn
import numpy as np
import wandb
import os
import gc
import random

import utils
import test


def train(cfg,
          model,
          optimizer,
          criterion,
          dataloader,
          val_dataloader,
          scheduler,
          ):
    
    print(f"optimizer : {optimizer}")
    print(f"criterion : {criterion}")
    
    # torch.autograd.set_detect_anomaly(True)
    # scaler = GradScaler()
    
    device = cfg['device']
    model = model.to(device)

    model.train()
    eval_unit = int(len(dataloader) // cfg['eval_interval'])
    save_unit = int(len(dataloader) // cfg['save_interval'])


    if cfg['gradient_scaler'] and cfg['mixed_precision']:
        scaler = torch.cuda.amp.GradScaler()


    for epoch in range(cfg['num_epochs']):
        loss_per_epoch = 0
        model.train()    

        for batch_id, (img, tgt, tgt_padding_mask) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{epoch} training...", ncols=100):
                
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
                    output = output.contiguous().view(-1, output.shape[-1])
                    gt_tgt = gt_tgt.contiguous().view(-1)
                    
                    loss = criterion(output, gt_tgt)
                    loss_per_epoch += loss.item()
                    
            else:
                output = model(img, input_tgt, 
                            tgt_padding_mask, tgt_mask)
                output = output.contiguous().view(-1, output.shape[-1])
                gt_tgt = gt_tgt.contiguous().view(-1)
                
                loss = criterion(output, gt_tgt)
                loss_per_epoch += loss.item()

            # output -> (B * max_len) x vocab_size
            
            if torch.isnan(loss):
                print(epoch, batch_id, "Nan Loss!!")
                print(epoch, batch_id, "Nan Loss!!")
                print(epoch, batch_id, "Nan Loss!!")
                # print(output)
                # print(gt_tgt)
                # print(output.max(), output.min())
                # print(torch.argmax(output))
                # print(torch.argmin(output))
                # print(gt_tgt.max(), gt_tgt.min())
                # print(loss)
                
                # del loss
                # del output
                # del gt_tgt
                
                # torch.cuda.empty_cache()
                # gc.collect()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            if isinstance(optimizer, list):
                for opt in optimizer:
                    opt.zero_grad()
            else:
                optimizer.zero_grad()
                
            if cfg['gradient_scaler'] and cfg['mixed_precision']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(), cfg['gradient_clip'])

            if cfg['gradient_scaler'] and cfg['mixed_precision']:
                if isinstance(optimizer, list):
                    for opt in optimizer:
                        scaler.step(opt)
                        scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if isinstance(optimizer, list):
                    for opt in optimizer:
                        opt.step()
                else:
                    optimizer.step()
            
            if scheduler:
                # print("scheduler_update")
                if isinstance(scheduler, list):
                    for sche in scheduler:
                        sche.step()
                else:
                    scheduler.step()
                # scheduler.step()
            
            if (batch_id+1) % cfg['print_interval'] == 0:
                wandb.log({'train loss':loss.item()})
                print(epoch, "train loss:", loss.item())
            
            if (batch_id+1) % save_unit == 0:
                # wandb.log({'train loss':(loss_per_epoch / (batch_id+1))})
                # print(epoch, "train loss:", loss_per_epoch/batch_id)
                utils.save_ckpt({'model': model.state_dict(),
                                 'optimizer': [opt.state_dict() for opt in optimizer],
                                 'scheduler': [sche.state_dict() for sche in scheduler],
                                 'seed': {
                                     'pytorch': torch.get_rng_state(),
                                     'pytorch_cuda': torch.cuda.get_rng_state(),
                                     'numpy': np.random.get_state(),
                                     'random': random.getstate()
                                    }
                                 },
                                cfg['save_dir'],
                                epoch+1,
                                int((batch_id+1) // save_unit))
                
            if (batch_id+1) % eval_unit == 0:
                test.evaluate(cfg, model, criterion, val_dataloader, cfg['PAD_idx'])
                model.train()
                  
            if cfg['eval_first'] and epoch == batch_id == 0:
                test.evaluate(cfg, model, criterion, val_dataloader, cfg['PAD_idx'])
                model.train()

# test eval
# wandb
        