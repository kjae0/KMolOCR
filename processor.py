from torchvision.transforms import transforms


class SmilesProcessor:
    def __init__(self, converter):
        self.converter = converter
        
    def __call__(self, x):
        return [self.converter[c] for c in x]
    

class PostProcessor:
    def __init__(self, converter):
        self.converter = converter
        
    def __call__(self, x):
        return [self.converter[c] for c in x]
    
    
def get_converter(df, k_col, v_col):
    converter = {}
    
    for i in range(len(df)):
        if df[k_col].iloc[i] not in converter:
            converter[df[k_col].iloc[i]] = df[v_col].iloc[i]
            
    return converter
           
def get_processors(cfg, vocab):
    img_processor = transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    preprocessor = SmilesProcessor(get_converter(vocab, "character", "number"))
    postprocessor = PostProcessor(get_converter(vocab, "number", "character"))
    
    return img_processor, preprocessor, postprocessor
