
# Standard Python libs
import pandas as pd
import numpy as np
from datetime import date, datetime
from time import mktime
import time
import random

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
# from tradingWithPython
import backtest as bt

import ipdb
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
    def __init__(self, symbols = ['BTC','ETH'], num_features = 7, batch_length = 1, drop = [0.33,0.33, 0.33]):
        num_curr = len(symbols)
        model = Sequential()
        model.add(LSTM(64,
                       input_shape=(num_curr, num_features),
                       return_sequences=True,
                       stateful=False))
        model.add(Dropout(drop[0]))

        model.add(LSTM(64,
                       return_sequences=True,
                       stateful=False))
        model.add(Dropout(drop[1]))
        model.add(LSTM(64,
                       return_sequences = True,
                       stateful = False))
        model.add(Dropout(drop[2]))
        model.add(Dense(4, kernel_initializer='lecun_uniform', activation = 'linear'))
        # model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        model.compile(loss='mse', optimizer='nadam', )
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
        self.X_train, self.X_test = self.split_data(prices, min(30, len(prices)), 0.2)
        # _,_ = self.init_state(self.X_train)
        # _,_ = self.init_state(self.X_test, )

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
        #print(self.symbols)
        for i, s in enumerate(self.symbols):
            close.append(X[s]['close'].values)
            diff = np.diff(close[i])
            diff = np.insert(diff, 0,0)
            sma15 = ta.SMA(X[s], timeperiod = 15)
            sma60 = ta.SMA(X[s], timeperiod = 60)
            rsi = ta.RSI(X[s], timeperiod = 14)
            atr = ta.ATR(X[s], timeperiod = 14)

            data.append(np.nan_to_num(np.vstack((close[i], diff, sma15, close[i]-sma15, sma15-sma60, rsi, atr))))
            # if not test:
            #     scaler = StandardScaler()
            #     scaler.fit(data[i])
            #     scaled_data = scaler.transform(data[i])
            #     ipdb.set_trace()
            #     data[i] = np.expand_dims(scaled_data, axis = 1)
            #     joblib.dump(scaler, '../data/{}scaler.pkl'.format(s))
            # elif test:
            #     scaler = joblib.load('../data/{}scaler.pkl'.format(s))
            #     # ipdb.set_trace()
            #     scaled_data = scaler.transform(data[i])
            data[i] = np.expand_dims(data[i], axis = 1)

        data = np.hstack(data).T
        close = np.vstack(close)
        state = data[0:self.batch_length,:,:]
        if test:
            self.data_test = data.copy()
        else:
            self.data = data.copy()
        return state.copy(), close.copy()


    def act(self, state, action, trades, step):
        '''
        Called ONLY from eval_Q function -- action variable is defined there
        INSERT DOCSTRING HERE
        '''
        if step + 2 == self.data.shape[0]:
            for i in range(len(self.symbols)):
                state[:,i,:] = self.data[step:step+1, i, :]
                term = True
                trades.iloc[step+1] = [0]*len(self.symbols)
            return state[np.newaxis, :], step, trades, term

        state = self.data[step+1, :, :]
        # Take trade action {0:Hold, 1:Buy, 2:Sell}
        for i in range(len(self.symbols)):
            # #ipdb.set_trace()
            if action[i] == 1:
                trades.iloc[step+1][i] == 100
            elif action[i] == 2:
                trades.iloc[step+1][i] == -100
            else:
                trades.iloc[step+1][i] == 0
        term = False
        step += 1
        return state[np.newaxis, :], step, trades, term

    def bestow(self, new_state, step, action, prices, trades, term, epoch = None , plot = False):
        '''
        INSERT DOCSTRING HERE
        '''

        bestowal = [0]*len(self.symbols)
        trades = trades.fillna(0)
        if term == False:
            for i in range(len(self.symbols)):

                b_test = bt.Backtest(pd.Series(data=prices[i,step-2:step], index = trades[step-2:step].index.values).astype(float),pd.Series(data = [j[i] for j in trades][step-2:step]).astype(float), signalType='shares')
                # ipdb.set_trace()
                bestowal[i] = ((b_test.data['price'].iloc[-1] -b_test.data['price'].iloc[-2]) * b_test.data['shares'].iloc[-1])
        else:
            for i in range(len(self.symbols)):
                # ipdb.set_trace()
                b_test = bt.Backtest(pd.Series(data=prices[i,:], index = trades.index.values), pd.Series(data = [j[i] for j in trades]) , signalType='shares')
                bestowal[i] = b_test.pnl.iloc[-1]
                if plot == True:
                    plt.figure(figsize = (6,8))
                    b_test.plotTrades()
                    plt.axvline(x=400, color='black', linestyle='--')
                    plt.text(250, 400, 'training data')
                    plt.text(450, 400, 'test data')
                    plt.suptitle(str(epoch))
                    plt.savefig('plt/'+str(epoch)+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
                    plt.close('all')
        return (bestowal)

    def eval_Q(self, eval_data, ep):
        '''
        INSERT DOCSTRING HERE
        '''
        trades = pd.Series(data = [np.array([0]*len(self.symbols)) for i in range(len(eval_data))],index=np.arange(len(eval_data)))
        state, prices = self.init_state(eval_data)
        state_val = state.copy()
        step = 1
        term = False
        go = True
        j = 0
        while go == True:
            #print(state_val.shape)
            q_values = self.rnn.predict(state_val.reshape(1,2,7))
            action = np.argmax(q_values, axis = 2)[0]
            state_prime, step, trades, term = self.act(state_val, action, trades, step)
            #ipdb.set_trace()
            bestowal = self.bestow(state_prime,step,action,prices, trades, term, epoch = ep)
            state = state_prime
            j += 1
            print(bestowal)
            if term:
                break
        return bestowal

    def fit(self, epochs = 100):
        '''
        INSERT DOCSTRING HERE
        '''
        #reinforcement learning parameters
        gamma = 0.95
        explore = 1
        mem_chunk = 50

        # "memory"
        buffer = 150
        h = 0
        st_mem = []

        learning_progress = [] #Stores state, Action, reward (bestowal :) ) and next state


        # reinforcement learning loop
        t0 = time.time()
        for i in range(epochs):
            # Set statespace for testing:
            if i == epochs - 1:
                state, close = self.init_state(self.X_test, test = True)
                t_data = self.data_test
            else:
                state, close = self.init_state(self.X_train)
                t_data = self.data
            go = True
            term = False
            trades = pd.Series(data = [np.array([0]*len(self.symbols)) for i in range(len(t_data))],index=np.arange(len(t_data)))
            # ipdb.set_trace()
            step = 3 # With a period of 4 hours (14400 sec), 3 periods = 12 hours

            while go:
                q_values = self.rnn.predict(state, batch_size = 1)
                # impliment exploration parameter!
                if (np.random.rand() < explore):
                    action = np.random.randint(4, size = 2)
                else:
                    action = np.argmax(q_values, axis = 2)[0]
                # take action and evaluate new state
                state_prime, step, trades, term = self.act(state, action, trades, step)
                # evaluate r (bestowal)
                bestowal = self.bestow(state_prime, step, action, close, trades, term)

                # Set up memory for reinforcement learning
                if len(st_mem) < buffer:
                    st_mem.append([state, action, bestowal, state_prime])
                # if memory is full, overwrite values
                else:
                    if h < buffer-1:
                        h += 1
                    else:
                        h = 0
                    # Throw some randomness up in here
                    st_mem[h] = [state, action, bestowal, state_prime]
                    batch = random.sample(st_mem, k = mem_chunk)
                    xTrain = []
                    yTrain = []
                    for mem in st_mem:
                        # ipdb.set_trace()
                        mem_state, mem_action, mem_bestowal, mem_state_prime = mem
                        mem_q = self.rnn.predict(mem_state)
                        max_mem_q = np.max(self.rnn.predict(mem_state_prime), axis = 2)
                        if term:
                            update = mem_bestowal
                        else:
                            update = mem_bestowal + (gamma * max_mem_q)
                        y = mem_q
                        # ipdb.set_trace()
                        for k in range(len(self.symbols)):
                            y[0][k][action[k]] = update[0][k]

                        xTrain.append(mem_state)
                        yTrain.append(y)
                    # ipdb.set_trace()
                    xTrain = np.squeeze(np.array(xTrain), axis = (1))
                    yTrain = np.squeeze(np.array(yTrain), axis = (1))
                    self.rnn.fit(xTrain,yTrain, batch_size = mem_chunk, epochs = 1, verbose = 1)
                    state = state_prime
                if term:
                    go = False
            ipdb.set_trace()
            epoch_bestowal = self.eval_Q(self.X_test, ep = 5)
            learning_progress.append(epoch_bestowal)

            # So we know what's going on.
            print('\n\nEPOCH: {} \nREWARD: {}\nEXPLORATION COEFFICENT: {}\n\n'.format(i, epoch_bestowal, explore))
            if explore > 0.1:
                explore -= 1.0/epochs
        eval_time = time.time() -t0

        print('Reinforcement Learning Completed in {} seconds'.format(eval_time))
        print(' -- {} seconds per Epoch\n'.format(eval_time/epochs))
        mod_to_pickle = self.rnn
        # joblib.dump(mod_to_pickle, '../data/RNN_model.pkl')
        for i, s in enumerate(self.symbols):
            b_test = bt.Backtest(pd.Series(data=close[i,:], index = trades.index.values), pd.Series(data = [j[i] for j in trades]) , signalType='shares')
            b_test.data['delta'] = b_test.data['shares'].diff().fillna(0)
            plt.figure(figsize = (12,8))
            plt.subplot(3,1,1)
            b_test.plotTrades()
            plt.subplot(3,1,2)
            b_test.pnl.plot(style = 'x-')
            plt.subplot(3,1,3)
            ipdb.set_trace()
            plt.plot(range(5), learning_progress)

            plt.savefig('../images/{}_summary.png'.format(s),bbox_inches = 'tight',pad_inches = 0.5, dpi = 72)


if __name__ == '__main__':
    coins = ['BTC','ETH']
    start = mktime(datetime(2015, 8,8).timetuple())
    end = time.time()
    model = Model(num_features = 7,symbols = coins)
    model.load_data(start = start, end = end, period = 14400, is_pkld = True)
    state, close = model.init_state(model.X_train)
    model.fit(epochs = 5)
