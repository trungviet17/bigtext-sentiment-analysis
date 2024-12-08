from kafka import KafkaConsumer
# from pymongo import MongoClient
from json import loads

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


for message in consumer:
    tweet = message.value[-1]  # get the Text from the list

    print("-> Tweet:", tweet)

    # # Prepare document to insert into MongoDB
    # tweet_doc = {
    #     "tweet": tweet,
    #     "prediction": class_index_mapping[int(prediction)]
    # }

    # # Insert document into MongoDB collection
    # collection.insert_one(tweet_doc)