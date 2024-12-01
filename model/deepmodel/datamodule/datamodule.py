from transformers import T5Tokenizer
from torch.utils.data import DataLoader 
import lightning as pl
from dataset import TwitterDataset


class TwitterDatamodule(pl.LightningDataModule): 

    def __init__(self, train_dir: str, valid_dir: str, tokenizer: T5Tokenizer, batch_size: int=32, input_max_len: int=25):
        super().__init__()

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.input_max_len = input_max_len
    
    def setup(self, stage=None):
        self.train_dataset = TwitterDataset(data_dir=self.train_dir, tokenizer=self.tokenizer, input_max_len=self.input_max_len)
        self.valid_dataset = TwitterDataset(data_dir=self.valid_dir, tokenizer=self.tokenizer, input_max_len=self.input_max_len)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    

if __name__ == '__main__': 
    pass 


    
