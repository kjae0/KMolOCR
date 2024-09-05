from collections import OrderedDict

import random
import config
import torch
import os
import numpy as np 


class CollateFnFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def get_collate_fn(self):
        def collate_fn(samples):
            images = []
            labels = []
            target_len = []
            
            for img, label, label_len in samples:
                images.append(img.unsqueeze(0))
                label = [self.cfg.sos_idx] + label + [self.cfg.eos_idx]
                labels.append(torch.LongTensor(label+[self.cfg.pad_idx for _ in range(self.cfg.max_len - len(label) + 1)]).unsqueeze(0))
                target_len.append(label_len)
                
            images = torch.concat(images, dim=0)
            labels = torch.concat(labels)
            label_len = torch.concat(target_len, dim=0)
            
            return images, labels, label_len
        return collate_fn

def make_pad_mask(tgt, pad_idx):
    mask = tgt == pad_idx
    return mask

def pack_padded_sequence(seq, cap_len):
    s = []
    
    for batch_id in range(len(seq)):
        s.append(seq[batch_id][:cap_len[batch_id]])
    
    return torch.concat(s)

def get_partial_state_dict(st, target):
    idx = 8+len(target)
    idx = len(target) + 1
    keys = st.keys()
    for k in [i for i in keys]:
        if target in k:
            continue
        else:
            st.pop(k)
    
    keys = st.keys()
    values = st.values()
    new_keys = []
    for key in keys:
        if target in key:
            new_key = key[idx:]    # remove the 'module.'
            new_keys.append(new_key)        
        else:
            new_key = key[7:]    # remove the 'module.'
            new_keys.append(new_key)
    new_dict = OrderedDict(list(zip(new_keys, values)))

    return new_dict

def save_ckpt(ckpt, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    torch.save(ckpt, os.path.join(save_dir, file_name))
    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # This setting can slow down the training process
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def eval_char_unit(prediction, ground_truth, pad_idx):
    crt, wrg = 0, 0
    
    match_mask = (prediction == ground_truth).long() * 10
    pad_mask = (ground_truth != pad_idx).long()
    
    match_mask += pad_mask
    crt += (match_mask == 11).sum()
    wrg += (match_mask == 1).sum()
    
    return crt, wrg
    
def eval_smiles_unit(prediction, ground_truth):
    crt, wrg = 0, 0
    
    match_mask = (prediction == ground_truth)
    match_mask = match_mask.all(dim=1)
    
    crt += match_mask.sum()
    wrg += len(match_mask) - match_mask.sum()
    
    return crt, wrg
