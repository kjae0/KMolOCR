from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import json
import os
import argparse
import random
import wandb

from datetime import datetime

from models import caption
import utils
import train
import schedulers
import dataset

def main(args):
    with open(args.cfg_dir, "r") as f:
        cfg = yaml.load(f, yaml.Loader)
    
    now = datetime.now()
    utils.seed_everything(cfg['seed'])
    cfg['save_dir'] = f"./{now.date()}_{now.hour}_{now.minute}_{cfg['seed']}"
    
    print(f"checkpoints save directory: {cfg['save_dir']}")
    
    with open(cfg['vocab_dir'], "r") as f:
        vocab = json.load(f)
        
    cfg['SOS_idx'] = vocab['<sos>']
    cfg['PAD_idx'] = vocab['<pad>']
    cfg['EOS_idx'] = vocab['<eos>']
    cfg['UNK_idx'] = vocab['<unk>']
    wandb.config.update(cfg)
    
    smiles_transform = transforms.Compose([
            torch.LongTensor
        ])

    img_transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dset = dataset.ImageSmilesDataset(img_dir=cfg['img_dir'],
                                    smiles_dir=cfg['smiles_dir'],
                                    vocab_dir=cfg['vocab_dir'],
                                    img_transform=img_transform,
                                    smiles_transform=smiles_transform,
                                    sos_idx=cfg['SOS_idx'],
                                    pad_idx=cfg['PAD_idx'],
                                    eos_idx=cfg['EOS_idx'],
                                    max_len=cfg['max_len'],
                                    test=cfg['test'],
                                    load_later=True)
    print("dataset loaded")
    train_images = dset.images[:int(len(dset)*0.9)]
    train_smiles = dset.smiles[:int(len(dset)*0.9)]
    test_images = dset.images[int(len(dset)*0.9):]
    test_smiles = dset.smiles[int(len(dset)*0.9):]
    train_dset = dataset.ImageSmilesContainer(train_images, train_smiles, dset.smiles_transform, 
                                              load_later=dset.load_later, img_transform=dset.img_transform, vocab=dset.vocab, max_len=dset.max_len)
    test_dset = dataset.ImageSmilesContainer(test_images, test_smiles, dset.smiles_transform, 
                                             load_later=dset.load_later, img_transform=dset.img_transform, vocab=dset.vocab, max_len=dset.max_len)
    cfg['vocab_size'] = len(dset.vocab)
    
    print(f"\nTrain dataset size: {len(train_dset)}")
    print(f"Test dataset size: {len(test_dset)}")
    
    model = caption.Caption(cfg)
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    decoder_optimizer = optim.AdamW(model.decoder.parameters(), lr=cfg['lr']/10, weight_decay=cfg['weight_decay'])
    encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=cfg['T_max'], eta_min=cfg['eta_min'])
    decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=cfg['T_max'], eta_min=cfg['eta_min'])
    # scheduler = schedulers.CosineAnnealingWarmUpRestarts(optimizer,
                                                        #  T_0=cfg['T_0'],
                                                        #  T_mult=cfg['T_mult'],
                                                        #  eta_max=cfg['eta_max'],
                                                        #  T_up=cfg['T_up'],
                                                        #  gamma=cfg['gamma']
                                                        #  )

    dataloader = DataLoader(train_dset,
                            batch_size=cfg['batch_size'],
                            shuffle=True,
                            drop_last=True,
                            num_workers=cfg['num_workers'],
                            )

    test_dataloader = DataLoader(test_dset,
                            batch_size=cfg['batch_size'],
                            shuffle=False,
                            drop_last=False,
                            num_workers=cfg['num_workers'],
                            )    
        
    model = model.to(cfg['device'])
    model = nn.DataParallel(model)
                
    # train.train(cfg,
    #             model=model,
    #             optimizer=optimizer,
    #             criterion=criterion,
    #             dataloader=dataloader,
    #             val_dataloader=test_dataloader,
    #             scheduler=scheduler)
    train.train(cfg,
                model=model,
                optimizer=[encoder_optimizer, decoder_optimizer],
                criterion=criterion,
                dataloader=dataloader,
                val_dataloader=test_dataloader,
                scheduler=[encoder_scheduler, decoder_scheduler])
        
    return None    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True)
    args = parser.parse_args()
    
    wandb.init(project="KMolOCR EfficientNet V2 Large Encoder")
    
    wandb.run.name = "0203_training"
    # wandb.run.name = "test"
    wandb.run.save()    
    # try:
    main(args)
    # except:
    #     wandb.finish()
    
