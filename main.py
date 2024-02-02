import torch
import torch.nn as nn

import os
import argparse

from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
from SwinOCSR import swinocsr, swin_transformerv2
from Efficientnet import efficientnet
from Image2SMILES import caption
from transformer.models.model import transformer

from torchvision.transforms import transforms

import train, train2
import utils
import dataset

def main(args):
    pad_idx = args.pad_idx
    sos_idx = args.sos_idx
    eos_idx = args.eos_idx
    max_len = args.max_len
    
    smiles_transform = transforms.Compose([
            torch.LongTensor
        ])

    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # transforms.ToTensor()
    ])

    dset = dataset.ImageSmilesDataset(img_dir="/mount/images/",
                                    smiles_dir="/mount/smiles/",
                                    vocab_dir="/mount/vocabulary.csv",
                                    img_transform=img_transform,
                                    smiles_transform=smiles_transform,
                                    sos_idx=sos_idx,
                                    pad_idx=pad_idx,
                                    eos_idx=eos_idx,
                                    max_len=max_len,
                                    test=1000,
                                    load_later=True)

    train_images = dset.images[:int(len(dset)*0.8)]
    train_smiles = dset.smiles[:int(len(dset)*0.8)]
    test_images = dset.images[int(len(dset)*0.8):]
    test_smiles = dset.smiles[int(len(dset)*0.8):]
    train_dset = dataset.ImageSmilesContainer(train_images, train_smiles, dset.smiles_transform, load_later=dset.load_later, img_transform=dset.img_transform)
    test_dset = dataset.ImageSmilesContainer(test_images, test_smiles, dset.smiles_transform, load_later=dset.load_later, img_transform=dset.img_transform)
    vocab_size = len(dset.df) + 3
    args.vocab_size = vocab_size
    args.i2c = dset.vocab_retrieve
    
    if args.mode == "SwinOCSR":
        args.d_model = 1536
        
        # backbone = swin_transformerv2.SwinTransformerV2().to(args.device)
        backbone = efficientnet.EfficientNetB3().to(args.device)
        
        seq2seq = transformer.Transformer(trg_pad_idx=pad_idx,
                                        trg_sos_idx=sos_idx,
                                        dec_voc_size=vocab_size,
                                        max_len=max_len,
                                        device=args.device,
                                        d_model=args.d_model).to(args.device)
        
        if args.multi_gpu:
            backbone = nn.DataParallel(backbone)
            seq2seq = nn.DataParallel(seq2seq)
        
        criterion = nn.CrossEntropyLoss()
        backbone_optimizer = optim.Adam(backbone.parameters(), lr=args.backbone_lr)  
        transformer_optimizer = optim.Adam(seq2seq.parameters(), lr=args.transformer_lr)
        backbone_scheduler = optim.lr_scheduler.CosineAnnealingLR(backbone_optimizer, T_max=300, eta_min=0)
        transformer_scheduler = optim.lr_scheduler.CosineAnnealingLR(transformer_optimizer, T_max=300, eta_min=0)
        dataloader = DataLoader(train_dset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=5,
                                collate_fn=utils.PadCollate(dim=0))

        test_dataloader = DataLoader(test_dset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=8,
                                collate_fn=utils.PadCollate(dim=0))        
        
        seq2seq = train.train(cfg=args,
                            backbone=backbone,
                            seq2seq=seq2seq,
                            backbone_optimizer=backbone_optimizer,
                            transformer_optimizer=transformer_optimizer,
                            criterion=criterion,
                            dataloader=dataloader,
                            val_dataloader=test_dataloader,
                            backbone_scheduler=backbone_scheduler,
                            transformer_scheduler=transformer_scheduler)
        utils.save_checkpoint(seq2seq, args.save_dir)
        
    elif args.mode == 'Image2SMILES':
        model = caption.Caption(args.device, 
                                vocab_size=vocab_size,
                                max_len=args.max_len-1,
                                encoder_model=args.encoder_model)
        print(f"Encoder is {args.encoder_model}")
        
        # state_dict = torch.load("/data/jaeyeong/chemocr_saved_models/Image2SMILES/29.pt")
        # decoder_state_dict = utils.get_partial_state_dict(state_dict, 'decoder')
        # state_dict = torch.load("/data/jaeyeong/chemocr_saved_models/Image2SMILES/29.pt")
        # ffn_state_dict = utils.get_partial_state_dict(state_dict, 'ffn')
        # model.decoder.load_state_dict(decoder_state_dict)
        # model.ffn.load_state_dict(ffn_state_dict)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RAdam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)

        dataloader = DataLoader(train_dset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=16,
                                # collate_fn=utils.PadCollate(dim=0)
                                )

        test_dataloader = DataLoader(test_dset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=8,
                                # collate_fn=utils.PadCollate(dim=0)
                                )    
        
        model = model.to(args.device)
        if args.multi_gpu:
            model = nn.DataParallel(model)
                
        train2.train(args,
                     model=model,
                     optimizer=optimizer,
                     criterion=criterion,
                     dataloader=dataloader,
                     val_dataloader=test_dataloader,
                     scheduler=scheduler)
        
    return None    
    
def boolean_string(s):
    if s == 'True':
        return True
    else:
        return False
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str)
    args.add_argument("--num_epochs", type=int, default=30)
    args.add_argument("--pad_idx", type=int, default=0)
    args.add_argument("--sos_idx", type=int, default=1)
    args.add_argument("--eos_idx", type=int, default=2)
    args.add_argument("--max_len", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=10)
    args.add_argument("--lr", type=float, default=3e-4)
    args.add_argument("--img_size", type=int, default=224)
    args.add_argument("--encoder_model", type=str, default="NFNet_F1")
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--save_dir", type=str, default="./effb3_trans.pt")
    args.add_argument("--multi_gpu", type=boolean_string, default=True)
    args = args.parse_args()
    main(args)
    
