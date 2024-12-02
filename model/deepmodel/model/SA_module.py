from lstm_model import SALSTM_Model
import lightning as pl 
import torch 
from transformers import BertTokenizer, TFBertForSequenceClassification
import torch.optim as optim 
from dataclasses import dataclass


@dataclass
class BertFinetunerConfig: 
    model_name: str = 'bert-base-uncased'

@dataclass 
class LSTMConfig: 
    vocab_size: int = 30522
    embedding_dim: int = 100
    hidden_dim: int = 256
    output_dim: int = 3
    n_layers: int = 2
    dropout: float = 0.5






class TwitterSentimentModel(pl.LightningModule): 

    def __init__(self, model,  tokenizer: BertTokenizer, lr: float=1e-3, ):
        super().__init__()

        self.model = model 
        self.tokenizer = tokenizer


    def forward(self, x):
        return self.model(x)


    

    def common_step(self): 



        pass


    def training_step(self): 
        pass


    def validation_step(self): 
        pass


    def configure_optimizers(self):
        return super().configure_optimizers()

    