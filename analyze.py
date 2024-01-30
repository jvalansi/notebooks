# import robin_stocks.robinhood as r
import sys
import json
import datetime
from tqdm.auto import tqdm

import boto3

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
    data = s3.get_object(Bucket=bucket_name, Key=".robinhood.json")
    trade_cred = json.loads(data['Body'].read().decode('utf8'))
    data = s3.get_object(Bucket=bucket_name, Key=".twilio.json")
    notify_cred = json.loads(data['Body'].read().decode('utf8'))
    
    n2r = News2ROI(openai_key, rapidapi_key, trade_cred, notify_cred)

    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # date = "2024-01-26"
    
    news = n2r.get_news(date)
    articles = news

    hours = 1
    # hours = 24*7
    articles = [article for article in articles if News2ROI.contains_words(article, words={'United States',"US"})]
    articles = [article for article in articles if n2r.is_recent(article['publishedAt']['date'], delta=datetime.timedelta(hours=hours))]
    
    res = []
    for article in tqdm(articles):
        res += [n2r.analyse_article(article, date)]

    threshold=3
    # threshold=4
    candidates = News2ROI.get_candidates(res, threshold=threshold)
    candidate = candidates.iloc[0].copy(deep=True)
    ticker = candidate['ticker']
    # ticker = 'aapl'
    option_data = n2r.get_option_data(ticker)
    option_df = News2ROI.parse_option_data(option_data)
    row = option_df[option_df['normalized_gain']==option_df['normalized_gain'].max()].iloc[0].to_dict()
    candidate['strike_price'] = row['strike_price']
    candidate['bid_price'] = row['bid_price']
    candidate['current_price'] = row['current_price']
    
    print(n2r.notify(candidate))

if __name__=="__main__":
    main(None, None)