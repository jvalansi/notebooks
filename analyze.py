# import robin_stocks.robinhood as r
import sys
import json
import datetime
from tqdm.auto import tqdm

import boto3
import pandas as pd

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

    df = News2ROI.load_candidates(bucket_name)
    
    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')
    # date = "2024-01-26"
    
    articles = n2r.get_news(date, source='google')
    print(len(articles))
    hours = 1
    # hours = 7
    articles = [article for article in articles if News2ROI.contains_words(article, words={'United States',"US"})]
    print(len(articles))
    articles = [article for article in articles if n2r.is_recent(article['publishedAt'], delta=datetime.timedelta(hours=hours))]
    print(len(articles))    

    analysis = [n2r.analyse_article(article) for article in tqdm(articles)]
    threshold=3
    # threshold=4
    buy_candidates = News2ROI.get_candidates(analysis, threshold=threshold)
    print("buy_candidates",buy_candidates)    
    if not buy_candidates.empty:
        candidate = buy_candidates.iloc[0].copy(deep=True)
        ticker = candidate['ticker']
        # ticker = 'aapl'
        option_data = n2r.get_option_data(ticker)
        option_df = News2ROI.parse_option_data(option_data)
        row = option_df[option_df['normalized_gain']==option_df['normalized_gain'].max()].iloc[0].to_dict()
        candidate['strike_price'] = row['strike_price']
        candidate['buy_price'] = candidate['bid_price'] = row['bid_price']
        candidate['stock_buy_price'] = candidate['current_price'] = row['current_price']
        candidate['date'] = pd.to_datetime(candidate['date'])

        print(n2r.notify(candidate))
    
        df.loc[-1] = candidate
        df = df.reset_index(drop=True)

    start = now - datetime.timedelta(hours=24)
    # start = now - datetime.timedelta(hours=72)
    end = now - datetime.timedelta(hours=23)
    sell_candidates = df[(start < df['date']) & (df['date'] < end)]
    print("sell_candidates", sell_candidates)
    if not sell_candidates.empty:
        sell = sell_candidates.iloc[0]
        sell_options = n2r.get_option_data(sell['ticker'])
        sell_options = News2ROI.parse_option_data(sell_options)
        row = sell_options.set_index('strike_price').loc[sell['strike_price']]
        df.loc[sell.name, 'sell_price'] = row['ask_price']
        df.loc[sell.name, 'stock_sell_price'] = row['current_price']
        df.loc[sell.name, 'roi'] = row['ask_price']/sell['bid_price']
        
        print(n2r.notify(df.loc[sell.name]))

    News2ROI.store_candidates(bucket_name, df)
    

if __name__=="__main__":
    main(None, None)