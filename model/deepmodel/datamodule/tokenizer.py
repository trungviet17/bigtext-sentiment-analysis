# tokenizer 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas as pd 

nltk.download('punkt')
nltk.download('stopwords')

class TwitterTokenizer: 
    def __init__(self, train_dir: str, val_dir: str, min_freq: int = 2):
        train_data = pd.read_csv(train_dir, header = None)
        val_data = pd.read_csv(val_dir, header = None)
        self.data = pd.concat([train_data, val_data], axis = 0 )
        
        
        self.stop_words = set(stopwords.words('english'))
        self.special_tokens = ['<sos>', '<eos>', '<unk>', '<pad>']

        
        self.min_freq = min_freq
        self.data_preprocess()
        self._build_vocab()


    def data_preprocess(self): 
        self.data.columns = ['ID', 'Entity', 'Sentiment', 'Text']   
        self.data.dropna(inplace = True)

    def _build_vocab(self): 

        self.vocab = set()
        self.vocab.update(self.special_tokens)
        for row in self.data['Text']: 
            tokens = word_tokenize(row.lower())
            filtered_tokens = [word for word in tokens if word.isalnum() ]
    
            freq_dist = FreqDist(filtered_tokens)        
            filtered_tokens = [word for word in filtered_tokens if freq_dist[word] >= self.min_freq]
            self.vocab.update(filtered_tokens)
         
        self.token2idx = {word: idx for idx, word in enumerate(list(self.vocab))}
        self.idx2token = {idx: word for idx, word in enumerate(list(self.vocab))}

        
    def encode(self, text: str, max_len: int):
        tokens = word_tokenize(text.lower())
        unk_id = self.token2idx['<unk>']
        encoded_seq = [self.token2idx.get(token, unk_id) for token in tokens]


        if len(encoded_seq) > max_len: 
            encoded_seq = encoded_seq[: (max_len - 2)]
        else: 
            encoded_seq += [self.token2idx['<pad>']] * (max_len - len(encoded_seq) - 2) 

        encoded_seq = [self.token2idx['<sos>']] + encoded_seq + [self.token2idx['<eos>']]
        return encoded_seq
    
    
    def decode(self, idxs: list): 

        tokens = [self.idx2token.get(idx, '<unk>') for idx in idxs]
        tokens = [token for token in tokens if token not in self.special_tokens]
        decoded_text = ' '.join(tokens)
        return decoded_text
    

if __name__ == '__main__': 

    def test(): 
        tokenizer = TwitterTokenizer('data/train.csv', 'data/val.csv', 5)
        print(tokenizer.vocab)
        print(tokenizer.token2idx)
        print(tokenizer.idx2token)

        text = 'I am coming to the borders and I will kill you...' 
        encoded = tokenizer.encode(text) 

        print(encoded)

        print(f'Original text: {text}')

        decoded = tokenizer.decode(encoded)
        print(f'Decoded text: {decoded}')

    test()