
# Standard Python libs
import pandas as pd
import numpy as np
from datetime import date
from time import mktime
import time

# Machine Learning Libs
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam, Nadam

#Trading libraries
import talib.abstract as ta
from poloniex import Poloniex
from tradingWithPython import backtest as bt
'''
Name:           Model

Author:         John Prendergast

Created:        June 6th, 2017

Requirements:

Numpy
Pandas
MatplotLib
scikit-learn
TA-Lib
|--Technical Analysis Library
Keras
Poloniex
tradingwithPython

'''

class Model(object):
    '''
    Initializes a class with the keras RNN model at the center.

    Inputs:

    - num_features: number of features to feed into the neural net (default : 7)
    - num_curr    : number of currency pairs to predict on (current default : 1)
    - drop        : proportion of neurons to drop in each Dropout layer [list]

    ----------------------------------------------------------------------------

    '''
    def __init__(self, num_features = 7, num_curr = 1, drop = [0.33, 0.33]):
        model = Sequential()
        model.add(LSTM(64,
                       input_shape=(1, num_features),
                       return_sequences=True,
                       stateful=False))
        model.add(Dropout(drop[0]))

        model.add(LSTM(64,
                       input_shape=(1, num_features),
                       return_sequences=False,
                       stateful=False))
        model.add(Dropout(drop[1]))

        model.add(Dense(4, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu')) #linear output so we can have range of real-valued outputs

        model.compile(loss='mse', optimizer='nadam')
        self.model = model

    def load_data(symbol, start, end, period, quantity, is_pkld = False):
        '''
        INSERT DOCSTRING HERE
        '''
        prices = pd.DataFrame.from_records(Poloniex().returnChartData(currencyPair = symbol, start = start, end = end, period = period), index = 'date')
        prices.index = pd.to_datetime(prices.index*1e9)
        self.X_train, self.X_test = self.split_data(prices, min(3000, len(prices)), 0.2)

        return self

    def split_data(data, quantity, test_size):
        '''
        INSERT DOCSTRING HERE
        '''
        return data.iloc[-quantity:-round(quantity*test_size),], data.iloc[-quantity:,]

    def init_state(X, test = False):
        '''
        INSERT DOCSTRING HERE
        '''
        close = X['close'].values
        diff = np.diff(close)
        diff = np.insert(diff, 0,0)
        sma15 = ta.SMA(X, timeperiod = 15)
        sma60 = ta.SMA(X, timeperiod = 60)
        rsi = ta.RSI(X, timeperiod = 14)
        atr = ta.ATR(X, timeperiod = 14)

        data = np.nan_to_num(np.vstack((close, diff, sma15, close-sma15, sma15-sma60, rsi, atr)))
        if not test:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            data = np.expand_dims(scaled_data, axis = 1)
            joblib.dump(scaler, '../data/scaler.pkl')
        elif test:
            scaler = joblib.load('../data/scaler.pkl')
            scaled_data = scaler.transform(data)
            data = np.expand_dims(scaled_data, axis = 1)
        data = data.T
        state = data[0:1,0:1,:]
        return state, data, close


    def act(state, data, action, trades, step):
        '''
        INSERT DOCSTRING HERE
        '''
        if step + 2 == data.shape[0]:
            state = data[step:step+1, 0:1, :]
            term = True
            trades.iloc[timestep+1] = 0

            return state, step, trades, term
        state = data[step:step+1, 0:1, :]
        # Take trade action {0:Hold, 1:Buy, 2:Sell}
        if action == 1:
            trades.iloc[step+1] == 1
        elif action == 2:
            trades.iloc[step+1] == -1
        else:
            trades.iloc[step+1 ==0]
        term = False
        step += 1
        return state, step, trades, term

    def bestow(new_state, step, action, data, trades, term, epoch = 0):
        '''
        INSERT DOCSTRING HERE
        '''
        bestowal = 0
        trades = trades.fillna(False)
        if term == False:
            b_test = bt.Backtest(pd.Series(data=[i for i in data[step-2:step]], index = trades[step-2:step].index.values),trades[step-2:step], signalType='shares')
            bestowal = ((b_test.data['price'].iloc[-1] -b_test.data['price'].iloc[-2]) * b_test.data['shares'].iloc[-1])
        else:
            b_test = bt.Backtest(pd.Series(data=[i for i in data], index = trades.index.values), trades, signalType='shares')
            bestowal = b_test.pnl.iloc[-1]
        return bestowal

    def eval_Q(eval_data, eval_model, epoch = 0):
        '''
        INSERT DOCSTRING HERE
        '''
        trades = pd.Series(index=np.arange(len(data)))
        state, data, prices = init_state(eval_data)
        step = 1
        term = False
        go = True
        while go:
            q_values = eval_model.predict(state, batch_size = 1)
            action = np.argmax(qval)
            state_prime, step, trades, term = act(state, data, action, trades, step)
            bestowal = bestow(state_prime,step,action,prices, trades, term, epoch = epoch)
            state = state_prime
            if not term:
                go == False
        return bestowal

    def fit(self, epochs = 100):
        '''
        INSERT DOCSTRING HERE
        '''
        epsilon = 1
        batchSize = 100
        buffer = 200
        replay = []
        learning_progress = []

        
