import os
import pandas as pd
import numpy as np
from poloniex import Poloniex
from sklearn.externals import joblib


def pull_articles(c = 'BTC', source = 'the-new-york-times'):
    news_api = os.environ['WSJ_API']
    c = currency
    currencies = joblib.load('../data/currencies.pkl')
    term = currencies[c]['name']
    'https://newsapi.org/v1/articles?source=the-next-web&sortBy=latest&apiKey={}'.format(news_api)
