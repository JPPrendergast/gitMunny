import numpy as np
import talib.abstract as ta
import pandas as pd
from datetime import date, datetime
from time import mktime
import time
import os
import sys
import pickle

from poloniex import Poloniex

class MarketData(object):
    '''
    Base Class for storing market data

    Parameters
    ----------
    start : datetime[s]
        timestamp for start date
    end   : datetime[s]
        timestamp for end date
    period: int
        number of seconds between data
        Allowable Values:
        -------------------
        | 5 Minutes  :  300   |
        |15 Minutes  :  900   |
        |30 Minutes  :  1800  |
        |   2 Hours  :  7200  |
        |   4 Hours  :  14400 |
        |  24 Hours  :  86400 |
    p_key : str
        Poloniex API key
    p_secret : str
        Poloniex secret token
    '''
    def __init__(self, start, end, period, init_cash = 1000, init_coins = 2):
        self.start = start
        self.end = end
        self.period = period
        self.p_key = os.environ['POLO_API']
        self.p_secret = os.environ['POLO_SECRET_KEY']
        self.init_cash = init_cash
        self.init_coins = init_coins


class BitcoinData(MarketData):
    '''
    Class for technical analysis and storage of bitcoin market data

    '''
    def __init__(self, start, end, period):
        super(BitcoinData, self).__init__(start, end, period)

    def pull_chart_data(self):
        start_year = pd.to_datetime(self.start, unit = 's', format = '%Y')
        end_year = pd.to_datetime(self.end, unit = 's', format = '%Y')
        try:
            prices = pickle.load('../data/{}-{}_BTC'.format(start_year, end_year))
        except:
            polo = Poloniex(key = self.p_key, secret = self.p_secret)
            prices = pd.DataFrame.from_records(polo.returnChartData(currencyPair = 'USDT_BTC', start = self.start, end = self.end, period = self.period), index = 'date')
            prices = prices.apply(lambda x: pd.to_numeric(x))
            pickle.dump(prices,'../data/{}-{}_BTC'.format(start_year, end_year))

        self.train = prices.iloc[0:int(len(prices)*0.7)]
        self.test = prices.iloc[int(len(prices)*0.7):]

        self.train_data = self.process_data(self.train)
        self.test_data = self.process_data(self.test)

    def process_data(self, X):
        close = X['close']
        diff = np.diff(close)
        diff = np.insert(diff, 0, 0)
        sma8 = ta.SMA(X, timeperiod = 8)
        sma28 = ta.SMA(X, timeperiod = 28)
        rsi = ta.RSI(X, timeperiod = 8)
        atr = ta.ATR(X, timeperiod = 8)
        return np.nan_to_num(np.vstack([close, diff, sma8, sma28, rsi, atr]))


    def random_train_data(self, num_prices):
        h = np.random.randint(len(train_data)-num_prices)
        out = self.train_data[h:h+num_prices]
        out[0,7] = self.init_cash
        out[0, 8] = out[0,7] + (self.init_coins * out[0,0])
        return out


    # @property
    # def close_train(self):
    #     return(self.train_data[:,0])
    #
    # @property
    # def close_test(self):
    #     return(self.test_data[:,0])






'''
placeholder
'''
