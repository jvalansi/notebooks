# import robin_stocks.robinhood as r
import sys
import json
import datetime
from tqdm.auto import tqdm

import boto3
from twilio.rest import Client

from news2roi import News2ROI

def main(event, context):
    s3 = boto3.client('s3')
    bucket_name = 'jvalansi-notebooks-data'
    data = s3.get_object(Bucket=bucket_name, Key=".aws.json")
    j = json.loads(data['Body'].read().decode('utf8'))
    openai_key = j['OPENAI_KEY']
    data = s3.get_object(Bucket=bucket_name, Key=".rapidapi.json")
    j = json.loads(data['Body'].read().decode('utf8'))
    rapidapi_key = j["key"]

    n2r = News2ROI(openai_key, rapidapi_key)

    date = datetime.datetime.today().strftime('%Y-%m-%d')
    date = "2024-01-12"
    
    news = n2r.get_news(date)


    us_news = [article for article in news if News2ROI.contains_words(article, words={'United States',"US"})]

    res = []
    for article in tqdm(us_news[:20]):
        res += [n2r.analyse_article(article, date)]

    def condition(x):
        try:
            return int(x['action'])<=3
        except (ValueError,TypeError):
            return False

    low = list(filter(condition, res))
    print(low[0])

    data = s3.get_object(Bucket=bucket_name, Key=".twilio.json")
    j = json.loads(data['Body'].read().decode('utf8'))

    account_sid = j['account_sid']
    auth_token = j['auth_token']
    client = Client(account_sid, auth_token)

    message = client.messages.create(
      from_=j['from'],
      body=json.dumps(low[0],indent=2),
      to=j['to']
    )

    print(message.sid)

if __name__=="__main__":
    main(None, None)