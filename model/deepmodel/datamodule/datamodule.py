from torch.utils.data import DataLoader, TensorDataset
import lightning as pl
from dataclasses import dataclass, asdict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import re 
from sklearn.preprocessing import LabelEncoder
import torch 


@dataclass
class TokenizerConfig:
    num_words: int = 10000
    ovv_token: str = '<OOV>'



@dataclass 
class TwitterDatamoduleConfig:
    train_dir: str
    valid_dir: str
    batch_size: int
    input_max_len: int
    tokenizer_config: TokenizerConfig

    def __post_init__(self):
        self.tokenizer = Tokenizer(**asdict(self.tokenizer_config))



class TwitterDatamodule(pl.LightningDataModule): 

    def __init__(self, train_dir: str, valid_dir: str, tokenizer: Tokenizer, batch_size: int=32, input_max_len: int=25):
        super().__init__()

        self.train_df = pd.read_csv(train_dir)
        self.val_df = pd.read_csv(valid_dir)


        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.input_max_len = input_max_len

        


    def preprocessing_data(self): 
        columns = ['ID', 'Category', 'Sentiment', 'Tweet']

        self.train_df.columns = columns
        self.val_df.columns = columns

        self.train_df.dropna(subset=['Tweet'], inplace=True)
        self.val_df.dropna(subset=['Tweet'], inplace=True)


        def clean_text(tweet): 
            tweet = tweet.lower()
            tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
            tweet = re.sub(r'@\w+', '', tweet)
            tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet)
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            return tweet

        self.train_df['Cleaned_Tweet'] = self.train_df['Tweet'].apply(clean_text)
        self.val_df['Cleaned_Tweet'] = self.val_df['Tweet'].apply(clean_text)

        self.train_df.drop_duplicates(subset=['Cleaned_Tweet'], inplace=True)
        self.val_df.drop_duplicates(subset=['Cleaned_Tweet'], inplace=True)


        self.X_train = self.train_df['Tweet'].fillna('').astype(str).values
        self.X_val = self.val_df['Tweet'].fillna('').astype(str).values

        self.y_train = self.train_df['Sentiment'].values
        self.y_val = self.val_df['Sentiment'].values


    def label_handling(self): 
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.y_train)
        self.y_val = label_encoder.transform(self.y_val)


    def text_handling(self): 
        self.tokenizer.fit_on_texts(self.X_train)

        self.X_train = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_val = self.tokenizer.texts_to_sequences(self.X_val)

        self.X_train = pad_sequences(self.X_train, maxlen=self.input_max_len, padding='post')
        self.X_val = pad_sequences(self.X_val, maxlen=self.input_max_len, padding='post')


    def setup(self, stage=None):
        self.preprocessing_data()
        self.label_handling()
        self.text_handling()

        self.X_train = torch.tensor(self.X_train, dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.X_val = torch.tensor(self.X_val, dtype=torch.long)
        self.y_val = torch.tensor(self.y_val, dtype=torch.long)


        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.valid_dataset = TensorDataset(self.X_val, self.y_val)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    

if __name__ == '__main__': 

    tokenizer_config = TokenizerConfig(num_words=10000, ovv_token='<OOV>')
    datamodule_config = TwitterDatamoduleConfig(train_dir='data/train.csv', valid_dir='data/valid.csv', batch_size=32, input_max_len=25, tokenizer_config=tokenizer_config)




    
