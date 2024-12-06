import onnxruntime
import numpy as np
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
import re
from sklearn.preprocessing import LabelEncoder

class SentimentPredictorONNX:
    def __init__(self, model_path, tokenizer, label_encoder, max_len=50):

        self.session = onnxruntime.InferenceSession(model_path)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_len = max_len

    def clean_tweet(self, tweet):
    
        tweet = tweet.lower()
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet)
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        return tweet

    def predict(self, tweet):
       
        cleaned_tweet = self.clean_tweet(tweet)
        sequence = self.tokenizer.texts_to_sequences([cleaned_tweet])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        input_data = np.array(padded_sequence, dtype=np.int64)

        
        inputs = {self.session.get_inputs()[0].name: input_data}
        logits = self.session.run(None, inputs)[0]

        predicted_class = np.argmax(logits, axis=1)[0]

        sentiment = self.label_encoder.inverse_transform([predicted_class])[0]
        return sentiment
    
if __name__ == '__main__': 


    label_encoder = LabelEncoder()
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

    predictor = SentimentPredictorONNX(
        model_path="lstm_classifier.onnx",  
        tokenizer=tokenizer,               
        label_encoder=label_encoder,        
        max_len=50                    
    )

    sample_tweet = "I absolutely love this product!"
    predicted_sentiment = predictor.predict(sample_tweet)
    print(f"Predicted Sentiment: {predicted_sentiment}")
