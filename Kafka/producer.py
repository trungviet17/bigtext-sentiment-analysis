from time import sleep
import csv 
from kafka import KafkaProducer
import json
import requests

url = "https://tiktok-api23.p.rapidapi.com/api/post/comments"

querystring = {"videoId":"7289041834003746056","count":"50","cursor":"0"}

headers = {
	"x-rapidapi-key": "b71ba4d679mshc84f69068042c6ep170194jsn74783a14e755",
	"x-rapidapi-host": "tiktok-api23.p.rapidapi.com"
}



topic = 'twitter'
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: 
                         json.dumps(x).encode('utf-8'))

response = requests.get(url, headers=headers, params=querystring)
response = response.json()['comments']
data = [comment['text'] for comment in response]
producer.send(topic, value=data)

for text in data: 
    producer.send(topic, value=text)
    sleep(3)

# with open('twitter_validation.csv') as file_obj:
#     reader_obj = csv.reader(file_obj)
#     for data in reader_obj: 
#         # print(data)
#         producer.send(topic, value=data)
#         sleep(3)