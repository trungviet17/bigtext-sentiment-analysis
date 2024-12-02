from transformers import BertTokenizer


class TwitterTokenizer: 

    def __init__(self, tokenizer: BertTokenizer, max_len: int=25, train_dir: str=None, valid_dir: str=None):
        
        self.tokenizer = tokenizer

    

    def _build_vocab(self): 
        pass 


    def encode(self, text: str, max_len: int=25):
        return self.tokenizer.batch_encode_plus([text], max_length=max_len, pad_to_max_length=True, return_tensors='pt', truncation=True)
    

    def decode(self, token):
        return self.tokenizer.decode(token)