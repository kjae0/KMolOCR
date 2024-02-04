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
        self.pad_idx = pad_idx
        self.max_len = max_len
        
        if test:
            self.img_dir_lst = [self.img_dir_lst[0]]
        
        for idx, img_d in tqdm(enumerate(self.img_dir_lst), desc=f"load images...", total=len(self.img_dir_lst), ncols=100):
            img_ds = os.listdir(os.path.join(img_dir, img_d))
            img_ds.sort()
            
            for d in img_ds:
                if load_later:
                    self.images.append(os.path.join(img_dir, img_d, d))
                else:
                    image = Image.open(os.path.join(img_dir, img_d, d)).convert("RGB")
                    image = img_transform(image)
                    self.images.append(image)

        for idx, smiles_d in tqdm(enumerate(self.smiles_dir_lst), desc="loading smiles...", total=len(self.smiles_dir_lst), ncols=100):
            df = pd.read_csv(os.path.join(smiles_dir, smiles_d))
            smiles = list(df['SMILES'])
            
            if test and len(self.smiles) > test:
                break
        
            if max_len:
                for i in range(len(smiles)):
                    if len(smiles[i]) > max_len - 2:
                        self.long_smiles_idx.append(i + len(self.smiles))
            
            self.smiles.extend(smiles)
            
        if test:
            self.images = self.images[:test]
            self.smiles = self.smiles[:test]
            
        if max_len:
            self.long_smiles_idx = self.long_smiles_idx[::-1]
            self.long_smiles_idx_set = set(self.long_smiles_idx)
            # self.long_smiles_idx.sort(reverse=True)
            new_smiles = []
            new_images = []
            for i in range(len(self.smiles)):
                try:
                    # del self.smiles[i]
                    # del self.images[i]
                    if i in self.long_smiles_idx_set:
                        continue
                    new_smiles.append(self.smiles[i])
                    new_images.append(self.images[i])
                except IndexError:
                    if test:
                        continue
                
            self.smiles = new_smiles
            self.images = new_images
                
        if len(self.images) != len(self.smiles):
            raise ValueError(f"difference image and smiles size. {len(self.images)} / {len(self.smiles)}")
                
        print(f"dataset size : {len(self.images)}")
        print(f"vocabulary size : {len(self.vocab)}")
        
    def __getitem__(self, idx):
        img = self.img_transform(Image.open(self.images[idx]).convert("RGB"))
        smiles = utils.smiles_processer(self.smiles[idx], self.max_len, self.sos_idx, self.eos_idx, self.pad_idx, self.vocab)
        smiles = self.smiles_transform(smiles)
        pad_mask = smiles == self.pad_idx
        return img, smiles, pad_mask
    
    def __len__(self):
        return len(self.images)

class ImageSmilesContainer(ImageSmilesDataset):
    def __init__(self, images, smiles, smiles_transform, load_later, vocab, max_len, img_transform=None):
        self.images = images
        self.smiles = smiles
        self.smiles_transform = smiles_transform
        self.load_later = load_later
        self.img_transform = img_transform
        self.pad_idx = vocab['<pad>']
        self.sos_idx = vocab['<sos>']
        self.eos_idx = vocab['<eos>']
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.images)
    
    
# data validation
# Nan
# data augmentation
# eval