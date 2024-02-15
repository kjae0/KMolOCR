from PIL import Image
from tqdm import tqdm

from torch.utils import data

import os
import pandas as pd
import torch


class ImageDataset(data.Dataset):
    def __init__(self, img_dir, 
                 smiles_dir,
                 img_transform=None,
                 smiles_transform=None,
                 max_len=100,
                 test=False):
        
        self.img_dir = img_dir
        self.smiles_dir = smiles_dir
        self.img_transform = img_transform
        self.smiles_transform = smiles_transform
        
        self.folders = os.listdir(self.img_dir)
        self.folders.sort()
        
        self.images = []
        self.smiles = []
        
        if test:
            self.folders = self.folders[:test]
        
        for folder in tqdm(self.folders, total=len(self.folders), desc="load dataset...", ncols=60):
            df = pd.read_csv(os.path.join(self.smiles_dir, folder+".csv"))
            images_fms = list(df['file_name'])
            
            # For dataset validation
            if len(images_fms) != len(df):
                raise ValueError(f"Expected same smiles csv DataFrame length and image files, but got images({len(images_fms)}) smiles({len(df)})")
            
            ext_imgs = []
            ext_smiles = []
            
            for i in range(len(df)):
                if len(df['SMILES'].iloc[i]) > max_len-1:
                    continue
                ext_imgs.append(images_fms[i])
                ext_smiles.append(df['SMILES'].iloc[i])
                    
            self.images.extend(ext_imgs)
            self.smiles.extend(ext_smiles)
            
        for i in self.smiles:
            if len(i) > 100:
                print(i)
            
        if len(self.images) != len(self.smiles):
            raise ValueError(f"Image and SMILES must have same size! got {len(self.images)}, {len(self.smiles)}")
            
    def __getitem__(self, idx):
        
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        tgt = self.smiles[idx]

        if self.img_transform:
            img = self.img_transform(img)
        if self.smiles_transform:
            tgt = self.smiles_transform(tgt)
            
        return img, tgt
    
    def __len__(self):
        return len(self.images)
    
    
class ImageContainer(ImageDataset):
    def __init__(self, images, smiles, origin_dset:ImageDataset):
        self.images = images
        self.smiles = smiles
        self.origin_dset = origin_dset
        
    def __getitem__(self, idx):
        try:
            img = Image.open(os.path.join(self.origin_dset.img_dir, self.images[idx])).convert("RGB")
            tgt = self.smiles[idx]
        except:
            print("\n", self.images[idx])
            img = Image.open(os.path.join(self.origin_dset.img_dir, self.images[0])).convert("RGB")
            tgt = self.smiles[0]
        
        # tgt_len = torch.LongTensor([100]).unsqueeze(0)
        tgt_len = torch.LongTensor([len(tgt) + 2]).unsqueeze(0)

        if self.origin_dset.img_transform:
            img = self.origin_dset.img_transform(img)
        if self.origin_dset.smiles_transform:
            tgt = self.origin_dset.smiles_transform(tgt)
            
        return img, tgt, tgt_len
    
    def __len__(self):
        return len(self.images)
        
    
def build_dataset(train_ratio, *args, **kwargs):
    origin_dset = ImageDataset(*args, **kwargs)
    split_idx = int(len(origin_dset) * train_ratio)
        
    train_dset = ImageContainer(images=origin_dset.images[:split_idx],
                                smiles=origin_dset.smiles[:split_idx],
                                origin_dset=origin_dset)

    test_dset = ImageContainer(images=origin_dset.images[split_idx:],
                               smiles=origin_dset.smiles[split_idx:],
                               origin_dset=origin_dset)
    
    return train_dset, test_dset
    

class ImageInferenceDataset(data.Dataset):
    def __init__(self, img_dir, 
                 image_files,
                 img_transform=None,
                 max_len=100,
                 test=False):
        
        self.img_dir = img_dir
        self.img_transform = img_transform
        
        self.images = image_files
        
        print(f"inference images : {len(self.images)}")
            
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")

        if self.img_transform:
            img = self.img_transform(img)
            
        return img
    
    def __len__(self):
        return len(self.images)