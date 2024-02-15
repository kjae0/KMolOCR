import torch

def filter_and_load_state_dict(model, state_dict):
    filtered_state_dict = {k.replace("module.decoder", ""): v for k, v in state_dict.items() if 'decoder' in k and 'embedding' not in k}    
    model.load_state_dict(filtered_state_dict, strict=False)
    return model
