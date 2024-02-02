from tqdm import tqdm
from torch import optim

import torch
import torch.nn as nn
import wandb
import os

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
    
    device = cfg['device']
    
    model.train()
    unit = int(len(dataloader) // 3)

    for epoch in range(cfg['num_epochs']):
        loss_per_epoch = 0
        model.train()    

        for batch_id, (img, tgt, tgt_padding_mask) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{epoch} training...", ncols=60):
            
            # img = torch.zeros((5, 3, 480, 480)).half().cuda()
# tgt = torch.zeros((5, 100)).long().cuda()
# tgt_padding_mask = torch.zeros((5, 100)).bool().cuda()
# tgt_mask = set_up_causal_mask(100, 'cuda').half()
            tgt_mask = utils.set_up_causal_mask(cfg['max_len'])
            if cfg['half']:
                img = img.half()
                tgt_mask.half()
                
            img = img.to(device)
            tgt = tgt.long().to(device)
            input_tgt = tgt[:, :-1].to(device)
            gt_tgt = tgt[:, 1:].to(device)
            tgt_padding_mask = tgt_padding_mask.bool().to(device)
            tgt_mask = tgt_mask.to(device)
            
            # output -> B x max_len x vocab_size
            output = model(img, input_tgt, 
                        tgt_padding_mask, tgt_mask)
            
            # output -> (B * max_len) x vocab_size
            output = output.contiguous().view(-1, output.shape[-1])
            gt_tgt = gt_tgt.contiguous().view(-1)
            
            loss = criterion(output, gt_tgt)
            loss_per_epoch += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                
            # if 0<batch_id<len(dataloader)-5 and batch_id % unit == 0:
            #     print(f"{epoch}/{cfg.num_epochs} - {batch_id}/{len(dataloader)} loss : {loss_per_epoch/len(dataloader)}\n")    
            #     # predictions = torch.concat(predictions, dim=0)
            #     # labels = torch.concat(labels, dim=0)
            #     _predictions = utils.retrieve_from_output(predictions, cfg.i2c, cfg.eos_idx)
            #     _labels = utils.retrieve_from_logit(labels, cfg.i2c, cfg.eos_idx)
            #     print(utils.accracy_score(_predictions, _labels))

            
        