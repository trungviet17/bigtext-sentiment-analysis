import torch.nn as nn 
import torch 
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class LSTMModelConfig: 
    vocab_size: int = 10000
    embedding_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 4

@dataclass
class SALSTMModelConfig: 
    vocab_size: int = 10000
    embedding_dim: int = 100
    hidden_dim: int = 256
    output_dim: int = 4
    n_layers: int = 2
    dropout: float = 0.5


@dataclass 
class BiLSTMModelConfig: 
    vocab_size: int = 10000
    embedding_dim: int = 100
    hidden_dim1: int = 128
    hidden_dim2: int = 64
    output_dim: int = 4 
    dropout_rate: float = 0.5 


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, vocab_size, embedding_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  
        out = self.fc(lstm_out)
        return out
    

class SALSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, 
                 n_layers: int, dropout: float): 

        super().__init__()
        self.output_dim = output_dim

        
        self.embedding_layers = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_model = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,
                                  num_layers=n_layers, dropout=dropout, batch_first=True)
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):

        embedded = self.dropout(self.embedding_layers(x))
        output , _ = self.lstm_model(embedded)

        attw = torch.softmax(self.attention(output), dim=1)
        context_vec = torch.sum(attw * output, dim=1)

        output = self.fc(self.dropout(context_vec))

        return output
    

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim :int =100, hidden_dim1 :int =128, 
                 hidden_dim2 :int =64, output_dim :int =4, dropout_rate=0.5):
        
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, bidirectional=True, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1 * 2)  # Sequence length
        self.dropout1 = nn.Dropout(dropout_rate)
    
        self.lstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, bidirectional=True, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim2 * 2)  
        self.dropout2 = nn.Dropout(dropout_rate)
    
        self.fc1 = nn.Linear(hidden_dim2 * 2, 64)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, output_dim)

        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.embedding(x) 
        x, _ = self.lstm1(x)  
        x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2) 
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)  
        x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2) 
        x = self.dropout2(x)
        x = x.mean(dim=1)  
        
        x = F.relu(self.fc1(x))  
        x = self.dropout3(x)
        x = self.fc2(x) 
        
        return x