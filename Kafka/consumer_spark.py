from kafka import KafkaConsumer
from json import loads
import pickle
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


topic = 'twitter'
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: loads(x.decode('utf-8')))

# Assuming you have a SparkSession already created
spark = SparkSession.builder \
    .appName("classify tweets") \
    .getOrCreate()

# Load the model
pipeline = PipelineModel.load("saved_models/logistic_regression_model.pkl")

class_index_mapping = { 0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant" }

for message in consumer:
    text_input = message.value[-1]
    data = [(text_input,),]  
    data = spark.createDataFrame(data, ["Text"])
    # Apply the pipeline to the new text
    processed_validation = pipeline.transform(data)
    prediction = processed_validation.collect()[0][6]

    print("-> Tweet:", text_input)

    print("-> Predicted Sentiment:", prediction)
    print("-> Predicted Sentiment classname:", class_index_mapping[int(prediction)])