from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch 

class TwitterDataset(Dataset): 

    def __init__(self, data_dir: str, tokenizer: BertTokenizer, input_max_len: int=25):
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

        source = self.tokenizer.batch_encode_plus([text], max_length=self.input_max_len, pad_to_max_length=True, return_tensors='pt', truncation=True)

        target = self.sentiment2idx[sentiment]

        return torch.tensor(source['input_ids']).squeeze(), torch.tensor(target)


    def _data_preprocess(self):
        self.dataframe.columns = ['ID', 'Entity', 'Sentiment', 'Text']
        self.dataframe.dropna(inplace=True)


if __name__ == '__main__':

    def test(): 
        pass 



    pass