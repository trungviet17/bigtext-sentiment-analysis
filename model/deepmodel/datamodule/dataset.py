from tokenizer import TwitterTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch 
import re 

class TwitterDataset(Dataset): 

    def __init__(self, data_dir: str, tokenizer: TwitterTokenizer, input_max_len: int=25):
        super().__init__()
        
        self.dataframe = pd.read_csv(data_dir)
        self.tokenizer = tokenizer
        self._data_preprocess()
        
        self.input_max_len = input_max_len
        self.sentiment2idx = {word : idx for idx, word in enumerate(list(self.dataframe['Sentiment'].unique()))}
        self.idx2sentiment = {idx : word for idx, word in enumerate(list(self.dataframe['Sentiment'].unique()))}



    def __len__(self): 
        return len(self.dataframe)


    def __getitem__(self, idx):
        text, sentiment = self.dataframe.loc[idx, 'Text'], self.dataframe.loc[idx, 'Sentiment']

        source = self.tokenizer.encode(text, self.input_max_len)

        target = self.sentiment2idx[sentiment]

        return torch.tensor(source, dtype = torch.long).squeeze(), torch.tensor(target, dtype = torch.long)


    def _data_preprocess(self):
        self.dataframe.columns = ['ID', 'Entity', 'Sentiment', 'Text']
        self.dataframe.dropna(inplace=True)
        self.dataframe = self.dataframe.reset_index(drop=True)






if __name__ == '__main__': 

    def test(): 
        tokenizer = TwitterTokenizer('data/train.csv', 'data/test.csv', 2)
        dataset = TwitterDataset('data/train.csv', tokenizer)
        print(dataset[0])

    test()

