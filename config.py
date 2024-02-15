import pandas as pd

vocab = pd.read_csv("/mount/vocabulary.csv")


CONFIG = {
    "vocab_dir" : "/mount/vocabulary.csv",
    "img_dir" : "/mount/images",
    "smiles_dir" : "/mount/smiles_cleansed",
    "max_len" : 100,
    "train_batch_size" : 16,
    "test_batch_size" : 48,
    "device" : "cuda",
    "encoder" : "efficientnet_b7",
    "num_epochs" : 9,
    "input_size" : 400,
    
    "encoder_lr" : 1e-3,
    "encoder_momentum" : 0.9,
    "encoder_weight_decay" : 1e-5,
    "encoder_nesterov" : True,
    "encoder_clipping" : 0.1,
    
    "decoder_lr" : 5e-3,
    "decoder_weight_decay" : 1e-5,
    
    "T_max" : 10000,
    "eta_min" : 0,
    
    "num_workers" : 32,
    "train_ratio" : 0.9,
    
    "sos_idx" : len(vocab),
    "pad_idx" : len(vocab)+1,
    "eos_idx" : len(vocab)+2,
    "test": False,
    
    
    "log_dir" : "/mount/0712_effb5_log" ,   
    "model_dir" : "/mount/model_ckpt/0712_effb5_model"
    # "log_dir" : "/mount/0626_f3_logs/",
    # "model_dir" : "/mount/0626_f3_saved_models/"
}
