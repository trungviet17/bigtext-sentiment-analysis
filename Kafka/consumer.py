from kafka import KafkaConsumer
# from pymongo import MongoClient
from json import loads
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from model.deepmodel.infer import SentimentPredictorONNX

# client = MongoClient('localhost', 27017)
# db = client['bigdata_project']
# collection = db['tweets']

topic = 'twitter'
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: loads(x.decode('utf-8')))


label_encoder = LabelEncoder()
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
model_dir = os.path.join(curr_dir, "model", "deepmodel", "model")

model_path = os.path.join(model_dir, "lstm_classifier.onnx")
predictor = SentimentPredictorONNX(
        model_path=model_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_len=50,
    )

for message in consumer:
    text_input = message.value[-1]  # get the Text from the list

    if text_input.strip():
        pred = predictor.predict(str(text_input))
    else:
        pred = "Unknown"
    
    print("-> Comment:", text_input)
    print("-> Sentiment:", pred)

    # # Prepare document to insert into MongoDB
    # comment_doc = {
    #     "comment": text_input,
    #     "prediction": pred
    # }

    # # Insert document into MongoDB collection
    # collection.insert_one(comment_doc)