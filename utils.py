from tqdm import tqdm

import torch

def set_up_causal_mask(seq_len):
    """Defines the triangular mask used in transformers.

    This mask prevents decoder from attending the tokens after the current one.

    Arguments:
        seq_len (int): Maximum length of input sequence
    Returns:
        mask (torch.Tensor): Created triangular mask
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask

def smiles_processer(smiles, max_len, sos_idx, eos_idx, pad_idx, vocab):
    ret = [sos_idx]
        
    for s in tqdm(smiles, total=len(smiles), ncols=60, desc="processing SMILES..."):
        if s in vocab:
            ret.append(vocab[s])
        else:
            ret.append(vocab['unk'])

    ret.append(eos_idx)
    ret.extend([pad_idx for _ in range(max_len-len(ret))])
    return ret
