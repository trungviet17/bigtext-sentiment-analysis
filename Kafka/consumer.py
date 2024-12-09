from kafka import KafkaConsumer
# from pymongo import MongoClient
from json import loads
import os
import pickle
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

project_root = os.path.dirname(os.path.abspath(__file__))

with open(project_root + "/model/deepmodel/model_cpt/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open(project_root + "/model/deepmodel/model_cpt/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


model_dir = os.path.join(project_root, "model", "deepmodel", "model")

model_path = os.path.join(model_dir, "bilstm_classifier.onnx")
predictor = SentimentPredictorONNX(
        model_path=model_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_len=50,
    )

    
pred = predictor.predict("hello world")

for message in consumer:
    text_input = message.value[-1]  # get the Text from the list

    if text_input.strip():
        pred = predictor.predict(text_input)
    else:
        pred = "Unknown"
    
    print("-> Comment:", text_input)
    print("-> Sentiment:", pred)
    print("----------------------------------------------")

    # # Prepare document to insert into MongoDB
    # comment_doc = {
    #     "comment": text_input,
    #     "prediction": pred
    # }

    # # Insert document into MongoDB collection
    # collection.insert_one(comment_doc)