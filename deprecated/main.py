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
    
    wandb.run.name = cfg['wandb_run_name']
    # wandb.run.name = "test"
    wandb.run.save()    
    
    now = datetime.now()
    utils.seed_everything(cfg['seed'])
    cfg['save_dir'] = f"./ckpt/{now.date()}_{now.hour}_{now.minute}_{cfg['seed']}"
    
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
    
    if cfg['model_configuration']['num_classifier_layer'] != 1:
        model._set_new_classifier(cfg['model_configuration']['num_classifier_layer'],
                                  cfg['model_configuration']['classifier_hidden_dim'],
                                  cfg['vocab_size'])
        
        print(f"Warning: Model classifier is replaced to new one. {cfg['model_configuration']['num_classifier_layer']} layers, {cfg['model_configuration']['classifier_hidden_dim']} dimension")
    
    model = model.to(cfg['device'])

    criterion = nn.CrossEntropyLoss()
    
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    decoder_optimizer = optim.AdamW(model.decoder.parameters(), lr=cfg['lr']/cfg['decoder_lr_factor'], weight_decay=cfg['weight_decay'])
    
    if cfg['scheduler'] == 'CosineAnnealing':
        encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=cfg['T_max'], eta_min=cfg['eta_min'])
        decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=cfg['T_max'], eta_min=cfg['eta_min'])
    elif cfg['scheduler'] == 'CosineAnnealingWarmUpRestarts':
        encoder_scheduler = schedulers.CosineAnnealingWarmUpRestarts(encoder_optimizer,
                                                                     T_0=cfg['T_0'],
                                                                     T_mult=cfg['T_mult'],
                                                                     eta_max=cfg['eta_max'],
                                                                     T_up=cfg['T_up'],
                                                                     gamma=cfg['gamma']
                                                                     )
        decoder_scheduler = schedulers.CosineAnnealingWarmUpRestarts(decoder_optimizer,
                                                                     T_0=cfg['T_0'],
                                                                     T_mult=cfg['T_mult'],
                                                                     eta_max=cfg['eta_max'] / cfg['decoder_lr_factor'],
                                                                     T_up=cfg['T_up'],
                                                                     gamma=cfg['gamma']
                                                                     )
    else:
        raise ValueError(f"Invalid scheduler! got {cfg['scheduler']}.")
                                                        
    optimizer = [encoder_optimizer, decoder_optimizer]
    scheduler = [encoder_scheduler, decoder_scheduler]
    
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
        
    model = nn.DataParallel(model)

    if cfg['state_dict_dir']:
        print(f"\nLoad state dict from {cfg['state_dict_dir']}")
        state_dict = torch.load(cfg['state_dict_dir'])
        model_state_dict = model.state_dict()
        # utils.generous_load_state_dict(model, state_dict['model'])
        
        unmatched_loaded_keys = [k for k in state_dict['model'].keys() if k not in model_state_dict]
        if unmatched_loaded_keys:
            print("Unmatched keys in the loaded state dict:", unmatched_loaded_keys)
        
        # Keys in the model's state dict but not in the loaded state dict
        unmatched_model_keys = [k for k in model_state_dict.keys() if k not in state_dict['model']]
        if unmatched_model_keys:
            print("Unmatched keys in the model's state dict:", unmatched_model_keys)
            
        model.load_state_dict(state_dict['model'], strict=False)
        
        if cfg['load_optimizer']:
            print("Optimizer and scheduler loaded from previous checkpoint.")
            for idx, opt_sd in enumerate(state_dict['optimizer']):
                optimizer[idx].load_state_dict(opt_sd)
                
            for idx, sche_sd in enumerate(state_dict['scheduler']):
                scheduler[idx].load_state_dict(sche_sd)
        # scheduler.load_state_dict(state_dict['scheduler'])
        else:
            print("State dict loaded except optimizer and scheduler.")
        
        seed = state_dict['seed']
        random.setstate(seed['random'])
        np.random.set_state(seed['numpy'])
        torch.set_rng_state(seed['pytorch'])
        torch.cuda.set_rng_state(seed['pytorch_cuda'])
        
    else:
        print(f"\nNo state dict found.")
                
    if cfg['partial_training']:
        if cfg['partial_training']['train_part'] == "encoder":
            print("Train partially (encoder).")
            for param in model.module.decoder.parameters():
                param.requires_grad = False
            optimizer = [optimizer[0]]
            scheduler = [scheduler[0]]
                
        elif cfg['partial_training']['train_part'] == "decoder":
            print("Train partially (decoder).")
            for param in model.module.encoder.parameters():
                param.requires_grad = False
            optimizer = [optimizer[1]]
            scheduler = [scheduler[1]]
            
        else:
            raise ValueError(f"Invalid training part! got {cfg['partial_training']['train_part']}.")
    else:
        print("\nTrain end-to-end.")
    
    train.train(cfg,
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
    
    wandb.init(project="KMolOCR EfficientNet V2 Large Encoder")
    
    # try:
    main(args)
    # except:
    #     wandb.finish()
    
