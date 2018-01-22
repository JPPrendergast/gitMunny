
# Standard Python libs
import pandas as pd
import numpy as np
from string import punctuation
from datetime import date, datetime
from time import mktime
import time
import random
import os
import sys
import json
from collections import deque

# Plotting and Viz libraries
import matplotlib.pyplot as plt
import matplotlib

# Machine Learning Libs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib

# Neural Network libs
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, Nadam

# Trading Libraries
import talib.abstract as ta
from poloniex import Poloniex
import backtest as bt

# misc. libraries
from twilio.rest import Client
import ipdb

# Globals
try:
    account = os.environ['TWILIO_API_KEY']
    twi_auth = os.environ['TWILIO_AUTH_TOKEN']
    to_num = os.environ['JOHN_NUM']
    from_num = os.environ['FROM_NUM']
    client = Client(account, twi_auth)
except:
    pass
chump = []
np.random.seed(42)

'''
Name:           Model

Author:         John Prendergast

Created:        June 6th, 2017

Last Modified: January 22nd, 2018

Requirements:

    Numpy
    Pandas
    MatplotLib
    scikit-learn
    TA-Lib
     --Technical Analysis Library
    Keras
    Poloniex
    tradingwithPython

For API:
    Docker
    Flask
    flask_restful

For text message updates
    Twilio


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

    def __init__(self, symbols=['BTC', 'ETH'], num_features=7, batch_length=1, drop=[0.33, 0.33, 0.33]):
        '''
        Initializes the Model object, and the recurrent neural net (brain) of the trading agent
        '''

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
        model.add(Dense(4, kernel_initializer='lecun_uniform', activation='linear'))
        model.compile(loss='mse', optimizer='sgd',
                      sample_weight_mode='temporal')
        self.rnn = model
        self.symbols = symbols
        self.batch_length = batch_length
        self.init_cash = 1000

    def load_data(self, quantity, start=None, end=None, period=14400, is_pkld=True, test=False):
        '''
        5 largest Cryptos by market volume = ['BTC', 'ETH', 'XRP', 'ETC', 'XMR']

        Method to download and initialize the data using Poloniex API
        quantity = sample size
        start, end, period = timestamp
        is_pkld = boolean, indicates whether the data has been pickled
        test = boolean, indicates testing epoch

        '''
        # api keys not necessary for gathering data, only for making trades
        try:
            polo_api = os.environ['POLO_API']
            polo_secret = os.environ['POLO_SECRET_KEY']
        except:
            polo_api = None
            polo_secret = None

        if is_pkld:
            scaled_prices = joblib.load(
                './data/prices_{}coins.pkl'.format(len(self.symbols)))
        else:
            sym_price = []
            a = []
            b = []
            for i in self.symbols:
                c = pd.DataFrame.from_records(Poloniex().returnChartData(
                    currencyPair='USDT_' + i, start=start, end=end, period=period), index='date')
                c.index = pd.to_datetime(c.index * 1e9)
                c = c.apply(lambda s: pd.to_numeric(s))
                sym_price.append(c)
                a.extend([i] * len(c.columns))
                b.extend(list(c.columns))

            # enforce uniform length
            length = min([len(i) for i in sym_price])
            sym_price = [i[-length:] for i in sym_price]

            # Form MultiIndexed DataFrame
            prices = pd.DataFrame(data=np.hstack(
                [i.values for i in sym_price]), index=sym_price[0].index, columns=pd.MultiIndex.from_tuples([i for i in zip(a, b)]))
            joblib.dump(
                prices, './data/prices_{}coins.pkl'.format(len(self.symbols)))
            scaled_prices = prices.copy()

        self.data_size = min(quantity, len(scaled_prices))

        # Split data into train/test sets
        self.X_train, self.X_test = self.split_data(
            scaled_prices, self.data_size, 0.2)

        if test:
            self.X_t = scaled_prices
        else:
            self.X_t = self.X_train.append(self.X_test)
        return self

    def split_data(self, data, quantity, test_size):
        '''
        Splits data according to quantity
        returns two numpy arrays
        '''
        if quantity < len(data):
            l = len(data) - quantity
            r = np.random.randint(l)
        else:
            quantity = len(data) - 1
            r = 0
        return data.iloc[r:int(quantity * (1 - test_size)) + r, ], data.iloc[r + int(quantity * (1 - test_size)):r + quantity]

    def init_state(self, X, test):
        '''
        Initializes state space for reinforcement learning
        '''
        data = []
        close = []

        for i, s in enumerate(self.symbols):
            # Process data using technical analysis library
            close.append(X[s]['close'])
            diff = np.diff(close[i])
            diff = np.insert(diff, 0, 0)
            sma15 = ta.SMA(X[s], timeperiod=14)
            sma60 = ta.SMA(X[s], timeperiod=56)
            rsi = ta.RSI(X[s], timeperiod=14)
            atr = ta.ATR(X[s], timeperiod=14)

            data.append(np.nan_to_num(
                np.vstack((close[i], diff, sma15, close[i] - sma15, sma15 - sma60, rsi, atr))))
            data[i] = np.expand_dims(data[i], axis=1)

        data = np.hstack(data).T
        close = np.vstack(close)
        state = data[0:1, :, :]
        if test:
            self.data_test = data
        else:
            self.data = data
        return state, close

    def predict_for_api(self, funds, risk=0.5):
        '''
        Method to call test, returns json indicating trade recommendation and
        price.
        '''
        end = time.time()
        start = end - (604800 / 32)
        period = 300
        self.load_data((end - start) / period, start=start,
                       end=end, period=period, is_pkld=False, test=True)
        q, trade, close = self.test()
        if trade == 1:
            p = 'Buy'
        elif trade == 2:
            p = 'Sell'
        else:
            p = 'Hold'

        if trade in (1, 2):
            # Determines value of trade based on quality of different decisions
            value = q.flatten()[trade] / sum(abs(q.flatten()))
        else:
            value = 0
        return json.dumps({"Datetime": end, "Trade": p, "Price": close, "Amount": "${}".format(round(value[0] * risk * float(funds), 2))})

    def act(self, indata, state, action, trades, step):
        '''
        Method to change state space given certain actions
        '''
        mask = np.array([0, 0.1, -0.1, 0])
        if step + 1 == indata.shape[0]:
            state = indata[step, :, :]
            state = np.expand_dims(state, axis=0)
            term = True
            trades[step] = [0] * len(self.symbols)
            return state, step, trades, term
        else:
            state = indata[step + 1: step + 2, :, :]
            # Take trade action {0:Hold, 1:Buy, 2:Sell}
            term = False
            trades[step] = mask[action]
            step += 1
            return state, step, trades, term

    def bestow(self, new_state, step, action, prices, trades, term, qs=None, epoch=None, plot=False, evaluate=False):
        '''
        returns reward for agent given the action taken
        '''
        bestowal = [0] * len(self.symbols)
        if evaluate == False:
            for i in range(len(self.symbols)):
                b_test = bt.Backtest(pd.Series(data=prices[i, step - 2:step], index=np.arange(step - 2, step)).astype(float), pd.Series(
                    data=trades[step - 2:step, i], index=np.arange(step - 2, step)).astype(float), roundShares=False, signalType='shares')
                bestowal[i] = ((b_test.data['price'].iloc[-1] -
                                b_test.data['price'].iloc[-2]) * b_test.data['trades'].iloc[-1])

        elif evaluate == True and term == True:
            for i in range(len(self.symbols)):
                close = self.data_test[:, 0, 0]

                b_test = bt.Backtest(pd.Series(data=prices[i, :], index=np.arange(len(trades))), pd.Series(
                    data=trades[:, i]), roundShares=False, initialCash=self.init_cash, signalType='shares')
                b_test2 = bt.Backtest(pd.Series(data=close, index=np.arange(len(trades))), pd.Series(
                    data=trades[:, i]), roundShares=False, initialCash=self.init_cash, signalType='shares')

                bestowal[i] = b_test.pnl.iloc[-1]
                if (plot == True) and ((epoch % 20 == 0) or epoch == 1):
                    matplotlib.rcParams['font.size'] = 24
                    f = plt.figure(figsize=(12, 12))
                    ax1 = f.add_subplot(2, 1, 1)
                    b_test2.plotTrades()
                    plt.axvline(x=round(self.data_size * 0.8),
                                color='black', linestyle='--')
                    ax2 = f.add_subplot(2, 1, 2)
                    ax2.plot(b_test2.data.netProfit,
                             color='green', label='Net Profit')
                    ax2.plot(b_test2.pnl, color='k', label='Adj. Profit')
                    plt.title('Profit and Loss')
                    plt.legend()

                    plt.savefig('./images/{}_{}sgd_summary.png'.format(
                        self.symbols[i], epoch), bbox_inches='tight', pad_inches=0.5, dpi=60)
                    plt.close('all')
                    self.init_cash = 1000
        self.init_cash = 1000
        return bestowal

    def eval_Q(self, eval_data, ep):
        '''
        evaluates quality of all actions taken
        '''
        bestowal = []
        state = eval_data[0:1, :, :]
        closeT = eval_data[:, :, 0]
        trades = np.array([np.zeros(len(self.symbols))] * len(eval_data))
        step = 1
        values = []
        actions = []
        term = False

        while not term:
            scaler = joblib.load('./data/close_standardScaler.pkl')
            state = scaler.transform(state[0, :, :])[np.newaxis, :, :]
            qs = self.rnn.predict_on_batch(state)
            action = np.argmax(qs, axis=2).flatten()
            values.append(qs)
            if step + 1 >= len(eval_data):
                term = True

            state_prime, step, trades, term = self.act(
                eval_data, state, action, trades, step)
            state = state_prime
            step + 1
            if term:
                bestowal = self.bestow(state_prime, step, action, closeT, trades,
                                       qs=values, term=term, epoch=ep, plot=term, evaluate=term)
                return np.array(bestowal)

    def fit(self, epochs=100):
        '''
        Method to train reinforcement learning.
        This is the big one.

        Gathers data, accumulates it into experience replay memory, and iteratively
        trains the neural network based on all the states determined by actions.

        Uses greedy-epsilon (Exploratory at first, exploitative after learning)
        '''

        # reinforcement learning parameters
        gamma = 0.95
        explore = 1
        mem_chunk = 8

        # "memory"
        buffer = 16
        h = 0
        st_mem = deque([])

        # Stores state, Action, reward (bestowal :) ) and next state
        self.learning_progress = []
        # reinforcement learning loop
        t0 = time.time()

        quantity = 96
        bar = 0
        state, p = self.init_state(self.X_train, test=False)
        stateT, pT = self.init_state(self.X_t, test=True)
        for i in range(epochs):
            print('%{} Completed'.format(i / epochs))
            l = np.random.randint(0, len(self.data) - quantity)

            # Set statespace for testing:
            if i == epochs - 1:
                self.load_data(np.inf, test=True)
                state, p = self.init_state(self.X_train, test=False)
                stateT, pT = self.init_state(self.X_t, test=True)
                trades = np.array(
                    [np.zeros(len(self.symbols))] * len(self.data_test))
                t_data = self.data_test
            else:
                trades = np.array([np.zeros(len(self.symbols))] * quantity)
                t_data = self.data[l:l + quantity]
            scaler = StandardScaler(with_mean=False)
            t_data = np.expand_dims(
                scaler.fit_transform(t_data[:, 0, :]), axis=1)
            joblib.dump(scaler, './data/close_standardScaler.pkl')

            state = t_data[0:1, :, :]
            close = p[:, l:l + quantity]

            go = True
            term = False

            step = 1
            rewards = []
            qVals = []

            while go:

                q_values = self.rnn.predict_on_batch(state)

                # implement exploration parameter!
                if (np.random.rand() < explore):
                    action = np.random.randint(4, size=len(self.symbols))
                else:
                    action = np.argmax(q_values, axis=2)[0]

                # eval new state space
                state_prime, step, trades, term = self.act(
                    t_data, state, action, trades, step)

                # evaluate reward (bestowal)
                bestowal = self.bestow(
                    state_prime, step, action, close, trades, term)

                if i == epochs - 1:
                    rewards.append(bestowal)

                # Set up memory for reinforcement learning
                if len(st_mem) < buffer:
                    st_mem.append([state, action, bestowal, state_prime])

                # if memory is full, overwrite values chronologically
                else:
                    # Throw some randomness up in here
                    st_mem.rotate(-1)
                    st_mem[-1] = [state, action, bestowal, state_prime]
                    k = np.random.randint(buffer - mem_chunk)
                    batch = list(st_mem)[k:k + mem_chunk]

                    xTrain = []
                    yTrain = []

                    for z, mem in enumerate(batch):
                        mem_state, mem_action, mem_bestowal, mem_state_prime = mem
                        mem_q = self.rnn.predict_on_batch(mem_state)
                        max_mem_q = np.max(
                            self.rnn.predict_on_batch(mem_state_prime), axis=2)
                        if term:
                            update = [mem_bestowal]
                        else:
                            update = mem_bestowal + (gamma * max_mem_q)
                        y = mem_q
                        for k in range(len(self.symbols)):
                            y[0][k][action[k]] = update[0][k]

                        xTrain.append(mem_state)
                        yTrain.append(y)

                    xTrain = np.squeeze(np.array(xTrain), axis=(1))
                    yTrain = np.squeeze(np.array(yTrain), axis=(1))

                    t1 = time.time()
                    self.rnn.train_on_batch(xTrain, yTrain)
                state = state_prime
                if term:
                    go = False

            # Take snapshot every 20 epochs
            if ((i + 1) % 20 == 0) or (i == 0):
                epoch_bestowal = self.eval_Q(self.data_test, ep=i + 1)
                self.learning_progress.append((i + 1, epoch_bestowal))
                print('\n\nEPOCH: {} \nREWARD: {}\nEXPLORATION COEFFICENT: {}\n\n'.format(
                    i + 1, epoch_bestowal, explore))
            else:
                pass

            '''
            message sends an update to my phone using Twilio api so I can keep track of it on the go
            '''
            # message = client.messages.create(to=to_num, from_=from_num, body = '\n\nEPOCH {} FINISHED\nREWARD: {}\nEXPLORATION COEFFICENT: {}\n\n'.format(i+1, epoch_bestowal[-1][-1], explore))

            # update exploration coefficient
            if explore > 0.1:
                explore -= (3 / epochs)
            else:
                explore = .01

            if (i % 10 == 0):
                # Save every 10 neural networks
                self.rnn.save('./data/model{}sgd.h5'.format(i), overwrite=True)

        eval_time = time.time() - t0

        # Print Reinforcement Learning summary
        print('Reinforcement Learning Completed in {} seconds'.format(
            round(eval_time, 2)))
        print(' -- {} seconds per Epoch\n'.format(round(eval_time / epochs, 2)))

        # Plot charts for further analysis of Reinforcement Learning session
        plt.figure(figsize=(12, 8))
        plt.plot(self.learning_progress[:, 0], self.learning_progress[:,
                                                                      1], label='Reward per Epoch', color='b', alpha=0.8)
        plt.plot(self.learning_progress[:, 0], moving_average(
            self.learning_progress[:, 1]), label='moving average', color='k')
        plt.savefig('reward_curve_sgd.png')
        plt.legend()
        plt.ylabel('Reward')
        plt.xlabel('Epoch')
        plt.close('all')

    def test(self):
        # load model here
        bestowal = []
        state, closeT = self.init_state(self.X_t, test=True)
        trades = np.array([np.zeros(len(self.symbols))] * len(self.data_test))

        values = []
        actions = []

        term = False
        scaler = joblib.load('./data/close_standardScaler.pkl')
        step = 1

        while not term:
            st = scaler.transform(state[0, :, :])[np.newaxis, :, :]

            qs = self.rnn.predict_on_batch(st)
            action = np.argmax(qs, axis=2)[0]

            values.append(qs)
            actions.append(action)

            if step + 1 == len(self.data_test):
                term = True
            state_prime, step, trades, term = self.act(
                self.data_test, state, action, trades, step)
            bestowal = self.bestow(
                state_prime, step, action, closeT, trades, term=True, evaluate=True)
            # print(bestowal)
            state = state_prime
            if term:
                return values[-1], actions[-1], closeT.flatten()[-1]


if __name__ == '__main__':
    # GOOD 28 mem chunk, 56 buffer, 100 epochs, 20 RNN epochs converged aroun 50
    # Bad: 14 mem chunk, 100 buffer, 200 epochs, 5 RNN epochs total shite
    # Current 21 mem chunk 21 buffer, 200 epochs, 50 RNN epochs also converges around 50

    coins = ['BTC']

    start = mktime(datetime(2013, 1, 1).timetuple())
    end = mktime(datetime(2017, 5, 30).timetuple())

    model = Model(num_features=7, symbols=coins)
    model.load_data(int((end - start) / 900), start=start,
                    end=end, period=900, test=False, is_pkld=True)
    state, close = model.init_state(model.X_train, test=False)
    client = Client(account, twi_auth)
    try:
        model.fit(epochs=20000)
        try:
            message = client.messages.create(
                to=to_num, from_=from_num, body="COMPUTE FINISHED -- SUCCESS")
        except:
            print("*-----TWILIO API KEYS NOT FOUND-----*\n---\nNo message sent\n---\n")
    except:
        '''
        Twilio api sends me errors should they be encountered
        '''
        try:
            message = client.messages.create(to=to_num, from_=from_num, body="COMPUTE FAILED \n Unexpected Error: {}".format(
                str(sys.exc_info()[0]).split()[1].strip(punctuation)))
            raise
        except:
            raise
