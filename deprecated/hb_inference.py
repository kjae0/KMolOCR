import torch
import torch.nn as nn
import pandas as pd

from model import caption

import processor

class Image2SmilesPredictor:
    def __init__(self, cfg:dict, build_model=True, multi_gpu=True):
        self.cfg = cfg
        self.multi_gpu = multi_gpu
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        
        self.i2c, self.c2i = self._build_vocabulary()
        
        if build_model:
            self.model = self._build_model()
            self.model = self.model.to(self.device)
        else:
            print("Model is not builded! Call 'set_model' manually.")
        
    def set_model(self, model):
        self.model = model
        
    def _build_model(self):
        vocab = pd.read_csv(self.cfg['vocab_dir'])
        model = caption.Caption(max_len=self.cfg['max_len'],
                                vocab_size=len(vocab)+3,
                                encoder_model=self.cfg['encoder']).to(self.device)
        
        if self.multi_gpu:
            model = nn.DataParallel(model)
        
        if "state_dict_dir" in self.cfg:
            self._load_state_dict(model, self.cfg['state_dict_dir'])
        else:
            print("Warning! key 'state_dict_dir' not found.\n")        
        
        print("Model and state dict loaded successfully.")
        
        return model
            
    def _load_state_dict(self, model, state_dict_dir):
        state_dict = torch.load(state_dict_dir)
        model.load_state_dict(state_dict)
        
    def _build_vocabulary(self):
        vocab = pd.read_csv(self.cfg['vocab_dir'])
        i2c = processor.get_converter(vocab, 'number', 'character')
        c2i = processor.get_converter(vocab, 'character', 'number')
        
        return i2c, c2i
        
    def _create_caption_and_mask(self, batch_size, start_token, max_length):
        caption_template = torch.zeros((batch_size, max_length), dtype=torch.long)
        mask_template = torch.ones((batch_size, max_length), dtype=torch.bool)
        caption_template[:, 0] = start_token
        mask_template[:, 0] = False
        
        return caption_template, mask_template 
    
    def _postprocessor(self, prediction: torch.Tensor):
        # prediction -> B x max_len
        processed = []
        
        for b in range(len(prediction)):
            truncated = prediction[b][prediction[b] != self.cfg['pad_idx']].tolist()
            
            pred = ""
            
            for c in truncated:
                if c == self.cfg['eos_idx']:
                    break
                
                if c in self.i2c:
                    pred += self.i2c[c]
                else:
                    pred += "<unk>"
                    
            processed.append(pred)            
            
        return processed
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        model.eval()
        
        if self.model == None:
            raise ValueError("Model not builded!")
        
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        
        result = []  
        for i in range(0, len(x), self.batch_size):
            try:
                x_pred = x[i:i+self.batch_size].to(self.device)
                
                caption, caption_mask = self._create_caption_and_mask(x_pred.shape[0], self.cfg['sos_idx'], self.cfg['max_len'])
                caption = caption.to(self.device)
                caption_mask = caption_mask.to(self.device)

                for pos in range(self.cfg['max_len'] - 1):
                    predictions = self.model(x_pred, caption, caption_mask)
                    predictions = predictions[:, pos, :]
                    predicted_id = torch.argmax(predictions, axis=-1)
                    
                    caption[:, pos+1] = predicted_id[0]
                    caption_mask[:, pos+1] = False
                    
                # caption -> B x max_len
                result.extend(self._postprocessor(caption.cpu()))
                        
            except:
                result.extend([None] * min(len(x)-i-1, self.batch_size))
            
        return result

