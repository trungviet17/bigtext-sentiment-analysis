import torch.nn as nn 
import torch 

class SALSTM_Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, 
                 n_layers: int, dropout: float): 

        super().__init__()
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