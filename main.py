import torch
import torch.nn as nn

import os
import argparse

from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.transforms import transforms
import yaml

from models import caption
import utils
import train
import dataset

def main(args):
    with open(args.cfg_dir, "r") as f:
        cfg = yaml.loa(f, yaml.Loader)
    
    smiles_transform = transforms.Compose([
            torch.LongTensor
        ])

    img_transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        # transforms.ToTensor()
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

    train_images = dset.images[:int(len(dset)*0.8)]
    train_smiles = dset.smiles[:int(len(dset)*0.8)]
    test_images = dset.images[int(len(dset)*0.8):]
    test_smiles = dset.smiles[int(len(dset)*0.8):]
    train_dset = dataset.ImageSmilesContainer(train_images, train_smiles, dset.smiles_transform, load_later=dset.load_later, img_transform=dset.img_transform)
    test_dset = dataset.ImageSmilesContainer(test_images, test_smiles, dset.smiles_transform, load_later=dset.load_later, img_transform=dset.img_transform)
    cfg['vocab_size'] = len(dset.df) + 3
    
    model = caption.Caption(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['T_max'], eta_min=cfg['eta_min'])

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
                
    train.train(args,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                dataloader=dataloader,
                val_dataloader=test_dataloader,
                scheduler=scheduler)
        
    return None    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
    
