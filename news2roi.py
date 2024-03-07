import os
import json
import random
import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser
import requests
from tqdm.auto import tqdm
from io import StringIO, BytesIO

import pytz
import pandas as pd
from openai import OpenAI
import robin_stocks.robinhood as r
from twilio.rest import Client
import boto3
from gnews import GNews

NO_TICKER = {"N/A","NONE","VARIES","VARIES","NOT LISTED","VARIOUS","NOT PUBLICLY TRADED","NOT PROVIDED","MULTIPLE","UNKNOWN","PRIVATE COMPANY","PRIVATE","NOT APPLICABLE","PRIVATE","UNAVAILABLE",","}
NUMERIC_COLS = ['strike_price','adjusted_mark_price','break_even_price', 'volume', 'chance_of_profit_long', 'ask_price', 'ask_size', 'bid_price', 'bid_size']

class News2ROI():
    def __init__(self, openai_key, news_key, trade_cred, notify_cred):
        os.environ["OPENAI_API_KEY"] = openai_key
        self.client = OpenAI()
        self.system_prompt = """
You are a financial analyst designed to provide a financial analysis of the short term impact of a certain stock for a given news article. Please provide the analysis in a JSON format.
The JSON should contain the following fields: a 'stock' field, a 'ticker' field, an 'exchange',  a 'reasoning' field and an 'action' field.
The 'stock' field is a united states traded stock related to the given news article.
The 'ticker' field is the ticker for that stock.
The 'exchange' field is the united states based stock exchange in which the stock is traded.
The 'reasoning' field is an expanation on how the given article affects the stock in the short term.
The 'action' field is the short term action that should be taken based on the reasoning on the article regarding the stock, it should be on a scale from 1 to 10, 1 being strong sell and 10 being strong buy.
"""
        self.news_key = news_key
        self.trade_cred = trade_cred
        self.notify_cred = notify_cred

    def rename_fields(self, d, article):
        article_ = {}
        for k,v in article.items():
            if k in d:
                article_[d[k]] = v
            else:
                article_[k] = v
        return article_

    def get_news(self, date=datetime.datetime.today().strftime('%Y-%m-%d'), params={}, source="reuters"):
        if source=="reuters":
            url = f"https://reuters-business-and-financial-news.p.rapidapi.com/article-date/{date}/0/20"
            headers = {
                "X-RapidAPI-Key": self.news_key,
                "X-RapidAPI-Host": "reuters-business-and-financial-news.p.rapidapi.com"
            }
            response = requests.get(url, headers=headers, params=params)
            articles = response.json()['articles']
            d = {"articlesName":"title","articlesShortDescription":"description"}
            articles = [self.rename_fields(d, article)  for article in articles]
        elif source=="google":
            articles = GNews().get_top_news()
            d = {"published date":"publishedAt"}
            articles = [self.rename_fields(d, article)  for article in articles]
            return articles
        else:
            payload['apiKey'] = self.news_key
            if country:
                payload['country'] = country
            url = f"https://newsapi.org/v2/everything"
            url = f"https://newsapi.org/v2/top-headlines"
            headers = {}

            response = requests.get(url, headers=headers, params=params)
            return response.json()

    def contains_words(article, words={'United States',"US"}):
        if 'keywords' not in article:
            return True
        keywords = set([k['keywordName'] for k in article['keywords']])
        tags = set([t['name'] for t in article['tags']])
        return bool(words.intersection(keywords)) or bool(words.intersection(tags))
    
    def is_recent(self, time, delta=datetime.timedelta( hours=1 )):
        try:
            published = parser.parse(time)
        except:
            published = parser.parse(time['date'])
        try:
            res = published >= datetime.datetime.now(tz=None) - delta
        except:
            res = published >= datetime.datetime.now(tz=pytz.UTC) - delta
        return res
    
    def is_after_hours(time):
        try:
            published = parser.parse(time)
        except:
            published = parser.parse(time['date'])
        return published.hour>=21
    
    def get_snp():
        import yfinance as yf

        GSPC = yf.Ticker("^GSPC")
        hist = GSPC.history(start="2014-01-01")
        hist['Diff'] = (hist['Close']-hist['Close'].shift(1))/hist['Close']
        hist = hist.reset_index()
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        snp = hist.set_index("Date")
        return snp
    
    def get_diff(ticker_name, date, after_hours):
        import yfinance as yf

        ticker = yf.Ticker(ticker_name)
        # duration = int(j['duration'])
        i = list(self.snp.index).index(date)
        if after_hours:
            start = date
            end = self.snp.iloc[i+2].name
        else:
            start = self.snp.iloc[i-1].name
            end = self.snp.iloc[i+1].name
        hist = ticker.history(start=start, end=end)
        if len(hist)==0:
            # print('no hist')
            return None,None,None

        ticker_start = hist.iloc[0]
        ticker_end = hist.iloc[-1]
        ticker_diff = (ticker_end['Close']-ticker_start['Close'])/ticker_start['Close']

        snp_start = self.snp.iloc[list(snp.index).index(start)]
        snp_end = self.snp.iloc[list(snp.index).index(end)]
        snp_diff = (snp_end['Close']-snp_start['Close'])/snp_start['Close']

        diff = ticker_diff-snp_diff

        return ticker_diff, snp_diff, diff

    def get_sentiment(self, article):
        response = self.client.chat.completions.create(
          model="gpt-4-1106-preview",
          response_format={ "type": "json_object" },
          messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"News article: {article}"},
          ]
        )

        return json.loads(response.choices[0].message.content)

    def analyse_article(self, article):
        data = f"{article['title']} - {article['description']}"
        if 'content' in article:
            data +=  " - {article['content']}"
        try:
            j = self.get_sentiment(data)
            if j['ticker'] in NO_TICKER or any([x in j['ticker'] for x in NO_TICKER]):
                # print('no ticker')
                return

            # ticker_diff, snp_diff, diff = get_diff(j['ticker'], date, is_after_hours(article['publishedAt']))
            published = article['publishedAt']
            j['date'] = published['date'] if 'date' in published else published
            j['data'] = data

            return j #, ticker_diff, snp_diff, diff
        except (TypeError, KeyError) as e:
            return

    def get_candidates(res, date=None, save=False, threshold=3):
        res = list(filter(None,res))
        df = pd.DataFrame(res)
        if df.empty:
            return df
        if save==True:
            df.to_csv(f"data/news/{date}.tsv", sep='\t')
        df['action'] = pd.to_numeric(df['action'], errors='coerce')
        low = df[df['action']<=threshold].sort_values('action')
        return low
    
    def get_option_data(self, ticker, retries=5):
        login = r.login(self.trade_cred['user'],self.trade_cred['pass'],expiresIn=60*60*24*30)
        today = datetime.date.today()
        days_till_next_friday = (4-today.weekday()) % 7
        if not days_till_next_friday:
            days_till_next_friday = 7
        friday = today + datetime.timedelta( days_till_next_friday )
        friday_s = friday.strftime("%Y-%m-%d")
        for _ in range(retries):
            optionData = r.find_options_by_expiration([ticker], expirationDate=friday_s,optionType='put')
            if len(optionData)>0:
                break
            today = friday
            friday = today + datetime.timedelta( 7 )
            friday_s = friday.strftime("%Y-%m-%d")
        return optionData
    
    def parse_option_data(optionData):
        option_df = pd.DataFrame(optionData)
        for col in NUMERIC_COLS:
            option_df[col] = pd.to_numeric(option_df[col])
        option_df['current_price'] = max(option_df['break_even_price'])
        option_df = option_df.sort_values('strike_price')
        option_df = option_df[(option_df['ask_size']>0) & (option_df['bid_size']>0)]
        option_df['pred_bid_price'] = option_df['bid_price'].shift(-1)
        option_df['gain'] = (option_df['pred_bid_price']-option_df['ask_price'])/option_df['ask_price']
        option_df['diff'] = option_df['strike_price'].shift(-1)-option_df['strike_price']
        option_df['normalized_gain'] = option_df['gain']/option_df['diff']
        option_df = option_df[['type']+ NUMERIC_COLS  + ['pred_bid_price'] + ['normalized_gain'] + ['current_price']]
        return option_df
    
    def notify(self, candidate):
        account_sid = self.notify_cred['account_sid']
        auth_token = self.notify_cred['auth_token']
        client = Client(account_sid, auth_token)

        message = client.messages.create(
          from_=self.notify_cred['from'],
          body=candidate.to_json(indent=2),
          to=self.notify_cred['to']
        )

        return message.sid
    
    def load_candidates(bucket_name):
        s3 = boto3.client('s3')
        data = s3.get_object(Bucket=bucket_name, Key="candidates.tsv")
        data = data['Body'].read().decode('utf8')
        df = pd.read_csv(StringIO(data), sep='\t', index_col=0)
        df['date'] = pd.to_datetime(df['date'])    
        return df

    def store_candidates(bucket_name, df):
        s3 = boto3.client('s3')
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        s_buf = BytesIO()        
        df.to_csv(s_buf, sep='\t', encoding='utf8')
        s_buf.seek(0)
        s3.put_object(Body=s_buf, Bucket=bucket_name, Key='candidates.tsv')


