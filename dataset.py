from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

import pandas as pd
import json
import utils
import os

def build_vocab(vocab):
    """
    vocab: json type, {"character": index}
    """
    
    vocab_c2i = vocab
    vocab_i2c = {}
    
    for k, v in vocab_c2i.items():
        vocab_i2c[v] = k
    
    return vocab_c2i, vocab_i2c


class ImageSmilesDataset(data.Dataset):
    def __init__(self, img_dir, 
                 smiles_dir, 
                 img_transform, 
                 smiles_transform, 
                 sos_idx, pad_idx, eos_idx, max_len, vocab_dir, 
                 test=None,
                 load_later=False):
        
        self.img_dir_lst = os.listdir(img_dir)
        self.smiles_dir_lst = os.listdir(smiles_dir)
        self.img_dir_lst.sort()
        self.smiles_dir_lst.sort()
        
        with open(vocab_dir, "r") as f:
            self.vocab = json.load(f)
        
        self.vocab, self.vocab_i2c = build_vocab(self.vocab)
        
        self.long_smiles_idx = []
        self.images = []
        self.smiles = []
        self.smiles_transform = smiles_transform
        self.img_transform = img_transform
        self.load_later = load_later
        
        if test:
            self.img_dir_lst = [self.img_dir_lst[0]]
        
        for idx, img_d in enumerate(self.img_dir_lst):
            img_ds = os.listdir(os.path.join(img_dir, img_d))
            img_ds.sort()
            
            for d in tqdm(img_ds, desc=f"load images from {img_d} ({idx+1}/{len(self.img_dir_lst)})", total=len(img_ds), ncols=60):
                if load_later:
                    self.images.append(os.path.join(img_dir, img_d, d))
                else:
                    image = Image.open(os.path.join(img_dir, img_d, d)).convert("RGB")
                    image = img_transform(image)
                    self.images.append(image)

        for idx, smiles_d in enumerate(self.smiles_dir_lst):
            df = pd.read_csv(os.path.join(smiles_dir, smiles_d))
            smiles = list(df['SMILES'])
            
            if test and len(self.smiles) > test:
                break
        
            if max_len:
                for i in range(len(smiles)):
                    if len(smiles[i]) > max_len - 2:
                        self.long_smiles_idx.append(i + len(self.smiles))
            
            smiles = [utils.smiles_processer(s, max_len, sos_idx, eos_idx, pad_idx, self.vocab) for s in smiles]
            self.smiles.extend(smiles)
            
        if test:
            self.smiles = self.smiles[:test]
            
        if max_len:
            self.long_smiles_idx.sort(reverse=True)
            for i in self.long_smiles_idx:
                try:
                    del self.smiles[i]
                    del self.images[i]
                except IndexError:
                    if test:
                        continue
                
        if len(self.images) != len(self.smiles):
            raise ValueError(f"difference image and smiles size. {len(self.images)} / {len(self.smiles)}")
                
        print(f"dataset size : {len(self.images)}")
        print(f"vocabulary size : {len(self.vocab)}")
        
    def __getitem__(self, index):
        # return self.images[index], self.smiles_transform(self.smiles[index])
        return transforms.ToTensor()(self.images[index]), self.smiles_transform(self.smiles[index])
    
    def __len__(self):
        return len(self.images)


class ImageSmilesContainer(data.Dataset):
    def __init__(self, images, smiles, smiles_transform, load_later, img_transform=None):
        self.images = images
        self.smiles = smiles
        self.smiles_transform = smiles_transform
        self.load_later = load_later
        self.img_transform = img_transform
        
    def __getitem__(self, index):
        if self.load_later:
            image = Image.open(self.images[index]).convert("RGB")
            image = self.img_transform(image)            
            return transforms.ToTensor()(image), self.smiles_transform(self.smiles[index])
        else:        
            return transforms.ToTensor()(self.images[index]), self.smiles_transform(self.smiles[index])
        # return self.images[index], self.smiles_transform(self.smiles[index])
    
    def __len__(self):
        return len(self.images)
    