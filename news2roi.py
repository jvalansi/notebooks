import os
import json
import random
import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser
import requests
from tqdm.auto import tqdm

from openai import OpenAI

NO_TICKER = {"N/A","NONE","VARIES","VARIES","NOT LISTED","VARIOUS","NOT PUBLICLY TRADED","NOT PROVIDED","MULTIPLE","UNKNOWN","PRIVATE COMPANY","PRIVATE","NOT APPLICABLE","PRIVATE","UNAVAILABLE",","}

class News2ROI():
    def __init__(self, openai_key, rapidapi_key):
        os.environ["OPENAI_API_KEY"] = openai_key
        self.client = OpenAI()
        self.system_prompt = """
You are a financial analyst designed to provide a financial analysis of the short term impact of a certain stock for a given news article. Please provide the analysis in a JSON format.
The JSON should contain the following fields: a 'stock' field, a 'ticker' field, a 'reasoning' field and an 'action' field.
The 'stock' field is a united states traded stock related to the given news article.
The 'ticker' field is the ticker for that stock.
The 'reasoning' field is an expanation on how the given article affects the stock in the short term.
The 'action' field is the short term action that should be taken based on the reasoning on the article regarding the stock, it should be on a scale from 1 to 10, 1 being strong sell and 10 being strong buy.
"""
        self.rapidapi_key = rapidapi_key
        

    def get_news(self, date):
        url = f"https://reuters-business-and-financial-news.p.rapidapi.com/article-date/{date}"

        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": "reuters-business-and-financial-news.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers)
        return response.json()

    def get_snp():        
        import yfinance as yf

        GSPC = yf.Ticker("^GSPC")
        hist = GSPC.history(start="2014-01-01")
        hist['Diff'] = (hist['Close']-hist['Close'].shift(1))/hist['Close']
        hist = hist.reset_index()
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        snp = hist.set_index("Date")
        return snp
    
    def contains_words(article, words={'United States',"US"}):
        keywords = set([k['keywordName'] for k in article['keywords']])
        tags = set([t['name'] for t in article['tags']])
        return bool(words.intersection(keywords)) or bool(words.intersection(tags))
    
    def is_recent(time, delta=datetime.timedelta( hours=1 )):
        published = parser.parse(time)
        return   published >= datetime.datetime.now() - delta
    
    def is_after_hours(article):
        article_date = article['publishedAt']['date']
        d = parser.parse(article_date)
        return d.hour>=21
    
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

    def analyse_article(self, article, date):
        data = f"{article['articlesName']} - {article['articlesShortDescription']}"
        try:
            j = self.get_sentiment(data)
            if j['ticker'] in NO_TICKER or any([x in j['ticker'] for x in NO_TICKER]):
                # print('no ticker')
                return

            # ticker_diff, snp_diff, diff = get_diff(j['ticker'], date, is_after_hours(article))
            j['date'] = date
            j['data'] = data

            return j #, ticker_diff, snp_diff, diff
        except (TypeError, KeyError) as e:
            return
