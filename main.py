from torch.utils.data import DataLoader

import argparse
import wandb
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from model import caption

import pandas as pd
import dataset
import train
import utils

def main(cfg, cfg_dict):
    vocab = pd.read_csv(cfg.vocab_dir)
    
    vocab['<sos>'] = cfg.sos_idx
    vocab['<pad>'] = cfg.pad_idx
    vocab['<eos>'] = cfg.eos_idx
    
    model = caption.Caption(max_len=cfg.max_len,
                            vocab_size=len(vocab)+3,
                            encoder_model=cfg.encoder).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)

    train_dset, test_dset = dataset.build_dataset(cfg=cfg,
                                                  vocab=vocab,
                                                  img_dir=cfg.img_dir,
                                                  smiles_dir=cfg.smiles_dir,
                                                  max_len=cfg.max_len,
                                                  train_ratio=cfg.train_ratio,
                                                  test=cfg.test)  
      
    collate_fn = utils.CollateFnFactory(cfg).get_collate_fn()
    train_dl = DataLoader(train_dset,
                          batch_size=cfg.train_batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=cfg.num_workers,
                          collate_fn=collate_fn)
    test_dl = DataLoader(test_dset,
                         batch_size=cfg.test_batch_size,
                         shuffle=False,
                         drop_last=False,
                          num_workers=cfg.num_workers,
                         collate_fn=collate_fn)

    model = nn.DataParallel(model).to(cfg.device)
    
    state_dict = torch.load(cfg.load_ckpt_dir)
    decoder_state_dict = utils.get_partial_state_dict(state_dict['weight'], 'module.decoder')
    model.module.decoder.load_state_dict(decoder_state_dict)
    
    print("model weight loaded successfully.")

    wandb.config.update(cfg_dict)
    
    if cfg.partial_training == "encoder":
        print("Train partially (encoder).")
        for param in model.module.decoder.parameters():
            param.requires_grad = False
                
    elif cfg.partial_training == "decoder":
        print("Train partially (decoder).")
        for param in model.module.encoder.parameters():
            param.requires_grad = False
            
    else:
        print("\nTrain end-to-end.")
    
    
    train.train(cfg=cfg,
                model=model,
                train_dl=train_dl,
                test_dl=test_dl,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler)
    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, required=True)
    args = parser.parse_args()
     
    with open(args.cfg_dir, "r") as f:
        cfg_dict = yaml.load(f, yaml.Loader)
    cfg = type("config", (object, ), cfg_dict)
    
    vocab = pd.read_csv(cfg.vocab_dir)
    cfg.sos_idx = len(vocab)
    cfg.pad_idx = len(vocab)+1
    cfg.eos_idx = len(vocab)+2
    
    wandb.init(project="KMolOCR EfficientNet V2 Large Encoder")
    wandb.run.name = cfg.wandb_run_name
    wandb.run.save()    
    
    main(cfg, cfg_dict)
