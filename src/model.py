
# Standard Python libs
import pandas as pd
import numpy as np
from datetime import date, datetime
from time import mktime
import time

# Plotting and Viz libraries
import matplotlib.pyplot as plt

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
    def __init__(self, symbols = ['BTC','ETH'], num_features = 7, num_curr = 1, batch_length = 14, drop = [0.33,0.33, 0.33]):
        model = Sequential()
        model.add(LSTM(64,
                       batch_input_shape=(batch_length,num_curr, num_features),
                       return_sequences=True,
                       stateful=True))
        model.add(Dropout(drop[0]))

        model.add(LSTM(64,
                       return_sequences=True,
                       stateful=True))
        model.add(Dropout(drop[1]))
        model.add(LSTM(64,
                       stateful = True))
        model.add(Dropout(drop[2]))
        model.add(Dense(4*len(symbols), kernel_initializer='lecun_uniform'))
        model.add(Activation('relu')) #linear output so we can have range of real-valued outputs

        model.compile(loss='mse', optimizer='nadam')
        self.rnn = model
        self.symbols = symbols
        self.batch_length = batch_length

    def load_data(self,  start = None, end = None, period = 14400, is_pkld = True):
        '''
        5 largest Cryptos by market volume = ['BTC', 'ETH', 'XRP', 'ETC', 'XMR']

        INSERT DOCSTRING HERE
        '''
        if is_pkld:
            prices = joblib.load('../data/prices_{}coins.pkl'.format(len(self.symbols)))
        else:
            sym_price = []
            a = []
            b = []
            for i in self.symbols:
                c = pd.DataFrame.from_records(Poloniex().returnChartData(currencyPair = 'USDT_' + i, start = start, end = end, period = period), index = 'date')
                c.index = pd.to_datetime(c.index*1e9)
                c = c.apply(lambda s: pd.to_numeric(s))
                sym_price.append(c)
                a.extend([i]*len(c.columns))
                b.extend(list(c.columns))
            # enforce uniform length
            length = min([len(i) for i in sym_price])
            sym_price = [i[-length:] for i in sym_price]

            # Form MultiIndexed DataFrame
            prices = pd.DataFrame(data = np.hstack([i.values for i in sym_price]), index = sym_price[0].index,columns = pd.MultiIndex.from_tuples([i for i in zip(a,b)]))
            joblib.dump(prices, '../data/prices_{}coins.pkl'.format(len(self.symbols)))
        self.X_train, self.X_test = self.split_data(prices, min(3000, len(prices)), 0.2)

        return self

    def split_data(self, data, quantity, test_size):
        '''
        INSERT DOCSTRING HERE
        '''
        return data.iloc[-quantity:-round(quantity*test_size),], data.iloc[-quantity:,]

    def init_state(self, X, test = False):
        '''
        INSERT DOCSTRING HERE
        '''
        data  = []
        close = []
        print(self.symbols)
        for i, s in enumerate(self.symbols):
            close.append(X[s]['close'].values)
            diff = np.diff(close[i])
            diff = np.insert(diff, 0,0)
            sma15 = ta.SMA(X[s], timeperiod = 15)
            sma60 = ta.SMA(X[s], timeperiod = 60)
            rsi = ta.RSI(X[s], timeperiod = 14)
            atr = ta.ATR(X[s], timeperiod = 14)

            data.append(np.nan_to_num(np.vstack((close[i], diff, sma15, close[i]-sma15, sma15-sma60, rsi, atr))))
            if not test:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[i])
                data[i] = np.expand_dims(scaled_data, axis = 1)
                print(data[i].shape)
                joblib.dump(scaler, '../data/{}scaler.pkl'.format(i))
            elif test:
                scaler = joblib.load('../data/{}scaler.pkl'.format(i))
                scaled_data = scaler.transform(data[i])
                data[i] = np.expand_dims(scaled_data, axis = 1)
                print(data[i].shape)

        data = np.hstack(data).T
        close = np.dstack(close)
        state = data[0:self.batch_length,:,:]
        return state, data, close


    def act(self, state, data, action, trades, step):
        '''
        Called ONLY from eval_Q function -- action variable is defined there
        INSERT DOCSTRING HERE
        '''
        if step + 2 == data.shape[0]:
            state = data[step:step+self.batch_length, :, :]
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
            trades.iloc[step+1] == 0
        term = False
        step += self.batch_length
        return state, step, trades, term

    def bestow(self, new_state, step, action, data, trades, term, epoch = 0, plot = False):
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
            if plot == True:
                plt.figure(figsize = (6,8))
                b_test.plotTrades()
                plt.axvline(x=400, color='black', linestyle='--')
                plt.text(250, 400, 'training data')
                plt.text(450, 400, 'test data')
                plt.suptitle(str(epoch))
                plt.savefig('plt/'+str(epoch)+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
                plt.close('all')
        return bestowal

    def eval_Q(self, eval_data, rnn, close, epoch = 0):
        '''
        INSERT DOCSTRING HERE
        '''
        trades = pd.Series(index=np.arange(len(eval_data)))
        state, data, prices = self.init_state(eval_data)
        step = 1
        term = False
        go = True
        while go:
            q_values = rnn.predict(state, batch_size = self.batch_length)
            action = np.argmax(q_values)
            state_prime, step, trades, term = self.act(state, data, action, trades, step)
            bestowal = self.bestow(state_prime,step,action,prices, trades, term, epoch = epoch)
            state = state_prime
            if not term:
                go == False
        return bestowal

    def fit(self, epochs = 100):
        '''
        INSERT DOCSTRING HERE
        '''
        pass
        # epsilon = 1
        # batchSize = 100
        # buffer = 200
        # replay = []
        # learning_progress = [] #Stores state, Action, reward (bestowal :) ) and next state
        #
        # trades = pd.Series(index = )

if __name__ == '__main__':
    coins = ['BTC','ETH']
    start = mktime(datetime(2015, 8,8).timetuple())
    end = time.time()
    model = Model(num_features = 7,symbols = coins, num_curr = len(coins))
    model.load_data(start = start, end = end, period = 14400, is_pkld = True)
    state, data, close = model.init_state(model.X_train, test = False)
    bestowal = model.eval_Q(model.X_train, model.rnn, close, epoch = 1)
