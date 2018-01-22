
# Standard Python libs
import pandas as pd
import numpy as np
from datetime import date, datetime
from time import mktime
import time
import random
import os
import sys
from twilio.rest import Client
# Plotting and Viz libraries
import matplotlib.pyplot as plt
import matplotlib
import json

# Machine Learning Libs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, Nadam
from string import punctuation
# Trading libraries
import talib.abstract as ta
from poloniex import Poloniex
# from tradingWithPython
import backtest as bt
# import progressbar
import ipdb
from collections import deque

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

    def __init__(self, symbols=['BTC', 'ETH'], num_features=7, batch_length=1, drop=[0.33, 0.33, 0.33]):
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
        # model.add(LSTM(16,
        #                return_sequences = True,
        #                stateful = False))
        # model.add(Dropout(drop[2]))
        model.add(Dense(4, kernel_initializer='lecun_uniform', activation='linear'))
        # model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        model.compile(loss='mse', optimizer='sgd',
                      sample_weight_mode='temporal')
        self.rnn = model
        self.symbols = symbols
        self.batch_length = batch_length
        self.init_cash = 1000

    def load_data(self, quantity, start=None, end=None, period=14400, is_pkld=True, test=False):
        '''
        5 largest Cryptos by market volume = ['BTC', 'ETH', 'XRP', 'ETC', 'XMR']

        INSERT DOCSTRING HERE
        '''
        #polo_api = os.environ['POLO_API']
        #polo_secret = os.environ['POLO_SECRET_KEY']
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
            # for s in self.symbols:
            #     for i in prices[s].columns:
            #         scaler = StandardScaler(with_mean = False)
            #         scaled_prices[s][i] = scaler.fit_transform(prices[s][i].reshape(-1,1)).flatten()
            #         joblib.dump(scaler, './data/{}_{}_standardScaler.pkl'.format(s, i))
            # joblib.dump(scaled_prices, './data/scaled_{}coins_prices.pkl'.format(s))
        self.data_size = min(quantity, len(scaled_prices))

        self.X_train, self.X_test = self.split_data(
            scaled_prices, self.data_size, 0.2)
        if test:
            self.X_t = scaled_prices
        else:
            self.X_t = self.X_train.append(self.X_test)
        return self

    def split_data(self, data, quantity, test_size):
        '''
        INSERT DOCSTRING HERE
        '''
        if quantity < len(data):
            l = len(data) - quantity
            r = np.random.randint(l)
        else:
            quantity = len(data) - 1
            r = 0
        # self.X_test_little = data.iloc(-((len(data)-quantity)-round(quantity*test_size)):)
        return data.iloc[r:int(quantity * (1 - test_size)) + r, ], data.iloc[r + int(quantity * (1 - test_size)):r + quantity]

    def init_state(self, X, test):
        '''
        INSERT DOCSTRING HERE
        '''
        data = []
        close = []
        # print(self.symbols)
        for i, s in enumerate(self.symbols):
            # import ipdb; ipdb.set_trace()
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
            value = q.flatten()[trade] / sum(abs(q.flatten()))
        else:
            value = 0
        # import ipdb; ipdb.set_trace()
        return json.dumps({"Datetime": end, "Trade": p, "Price": close, "Amount": "${}".format(round(value[0] * risk * float(funds), 2))})

    def act(self, indata, state, action, trades, step):
        '''
        INSERT DOCSTRING HERE
        '''
        # print(action)
        mask = np.array([0, 0.1, -0.1, 0])
        # print(step, indata.shape[0])
        if step + 1 == indata.shape[0]:
            state = indata[step, :, :]
            state = np.expand_dims(state, axis=0)
            term = True
            trades[step] = [0] * len(self.symbols)
            return state, step, trades, term
        else:
            state = indata[step + 1: step + 2, :, :]
            # Take trade action {0:Hold, 1:Buy, 2:Sell}
            # print(trades)
            # for i in range(len(self.symbols)):
            term = False
            try:
                trades[step] = mask[action]
            except ValueError:
                import ipdb
                ipdb.set_trace()

                # import ipdb; ipdb.set_trace()
            step += 1
            # print(step)
            return state, step, trades, term

    def bestow(self, new_state, step, action, prices, trades, term, qs=None, epoch=None, plot=False, evaluate=False):
        '''
        INSERT DOCSTRING HERE
        '''
        bestowal = [0] * len(self.symbols)
        if evaluate == False:
            for i in range(len(self.symbols)):
                # ipdb.set_trace()
                b_test = bt.Backtest(pd.Series(data=prices[i, step - 2:step], index=np.arange(step - 2, step)).astype(float), pd.Series(
                    data=trades[step - 2:step, i], index=np.arange(step - 2, step)).astype(float), roundShares=False, signalType='shares')
                bestowal[i] = ((b_test.data['price'].iloc[-1] -
                                b_test.data['price'].iloc[-2]) * b_test.data['trades'].iloc[-1])

        elif evaluate == True and term == True:
            for i in range(len(self.symbols)):
                # ipdb.set_trace()
                # scaler = joblib.load('./data/close_standardScaler.pkl')
                # close = scaler.inverse_transform(prices[i,:].reshape(-1,1)).flatten()
                close = self.data_test[:, 0, 0]

                b_test = bt.Backtest(pd.Series(data=prices[i, :], index=np.arange(len(trades))), pd.Series(
                    data=trades[:, i]), roundShares=False, initialCash=self.init_cash, signalType='shares')
                # bestowal[i] = ((b_test.data['price'].iloc[-1] -b_test.data['price'].iloc[-2]) * b_test.data['shares'].iloc[-2])
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
                    # plt.text(250, 400, 'training data')
                    # plt.text(450, 400, 'test data')
                    ax2 = f.add_subplot(2, 1, 2)
                    ax2.plot(b_test2.data.netProfit,
                             color='green', label='Net Profit')
                    ax2.plot(b_test2.pnl, color='k', label='Adj. Profit')
                    plt.title('Profit and Loss')
                    plt.legend()
                    # plt.subplot(6,1,3,sharex = ax)
                    # b_test2.data.value.plot(label = 'value')
                    # b_test2.data.cash.plot(label = 'cash')
                    # plt.title('Cash and Coin')
                    # plt.legend()
                    # plt.subplot(6,1,4,sharex = ax)
                    # for y, label in zip(np.array(qs)[:,0,0,:].T, ['Hold','Buy','Sell','Hold']):
                    #     plt.plot(y, label = label) #/np.vstack([np.array(qs)[:,0,0,:].sum(axis = 1)]*4).T
                    # plt.legend(loc = 'best')
                    # plt.title('Q-Values')
                    # plt.subplot(6,1,5,sharex = ax)
                    # plt.plot(np.argmax(np.array(qs)[:,0,0,:], axis = 1))
                    # plt.title('Max Q-value')
                    # plt.subplot(6,1,6,sharex = ax)
                    # b_test2.data.shares.plot()
                    # plt.title('Total Coins')
                    # plt.legend()
                    # plt.suptitle(self.symbols[i]+ ': Epoch --' + str(epoch))
                    plt.savefig('./images/{}_{}sgd_summary.png'.format(
                        self.symbols[i], epoch), bbox_inches='tight', pad_inches=0.5, dpi=60)
                    plt.close('all')
                    self.init_cash = 1000
                # print(trades)
        self.init_cash = 1000
        # joblib.dump(b_test.data, './data/epoch_{}_backtest.pkl'.format(epoch))
        return bestowal

    def eval_Q(self, eval_data, ep):
        '''
        INSERT DOCSTRING HERE
        '''
        bestowal = []
        # state, closeT = self.init_state(eval_data, test = True)
        state = eval_data[0:1, :, :]
        closeT = eval_data[:, :, 0]
        trades = np.array([np.zeros(len(self.symbols))] * len(eval_data))
        step = 1
        values = []
        actions = []
        term = False
        # import ipdb; ipdb.set_trace()

        while not term:
            scaler = joblib.load('./data/close_standardScaler.pkl')
            state = scaler.transform(state[0, :, :])[np.newaxis, :, :]
            qs = self.rnn.predict_on_batch(state)
            action = np.argmax(qs, axis=2).flatten()
            values.append(qs)

            # actions.append(action)
            if step + 1 >= len(eval_data):
                term = True

            state_prime, step, trades, term = self.act(
                eval_data, state, action, trades, step)
            state = state_prime
            step + 1
            if term:
                bestowal = self.bestow(state_prime, step, action, closeT, trades,
                                       qs=values, term=term, epoch=ep, plot=term, evaluate=term)
                # import ipdb; ipdb.set_trace()
                return np.array(bestowal)

        # trades = np.array([np.zeros(len(self.symbols))]*len(self.data_test))
        # # trades = pd.Series(data = [np.zeros(len(self.symbols))]*len(self.data_test), index = np.arange(len(self.data_test)))
        # state, prices = self.init_state(eval_data, test = True)
        # state_val = state
        # step = 3
        # term = False
        # go = True
        # j = 0
        # while go == True:
        #     #print(state_val.shape)
        #     q_values = self.rnn.predict_on_batch(state_val.reshape(1,2,7))
        #     action = np.argmax(q_values, axis = 2)
        #     state_prime, step, trades, term = self.act(self.data_test, state_val, action, trades, step)
        #
        #     bestowal = self.bestow(state_prime,step,action,prices, trades, term,evaluate = True, epoch = ep)
        #     state = state_prime
        #     j += 1
        #     # print(bestowal)
        #     if term:
        #         return np.array(bestowal)

    def fit(self, epochs=100):
        '''
        INSERT DOCSTRING HERE
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
        # bar =  progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') '], max_value = epochs * quantity).start()
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
                # close = pT
                # state = stateT
            else:
                trades = np.array([np.zeros(len(self.symbols))] * quantity)
                # trades = pd.Series(data = [[0]*(len(self.symbols))]*len(self.data), index = np.arange(len(self.data)))
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
            # state = t_data[i:i+1,:,:]
            # close = t_data
            # if i == 0:
            #     scaler = StandardScaler()
            #     qvs = self.rnn.predict_on_batch(t_data)
            #     scaler.fit(qvs[:,0,:])
            # print('RL Epoch {}'.format(i))
            rewards = []
            # import ipdb; ipdb.set_trace()
            qVals = []
            while go:
                q_values = self.rnn.predict_on_batch(state)

                # impliment exploration parameter!
                if (np.random.rand() < explore):
                    action = np.random.randint(4, size=len(self.symbols))
                else:
                    # if abs(q_values[:,:,1] - q_values[:,:,2]) < abs(q_values[:,:,2]*0.1):
                    #     action = np.zeros(len(self.symbols)).astype(int)
                    # else:
                    #     action = [np.zeros(len(self.symbols)).astype(int),np.argmax(q_values, axis = 2)[0,:]][np.random.choice(2)]

                    action = np.argmax(q_values, axis=2)[0]
                    # print(q_values[:,:,action]/q_values.sum())
                # take action and evaluate new state
                # print(action, np.argmax(q_values, axis = 2))
                state_prime, step, trades, term = self.act(
                    t_data, state, action, trades, step)
                # print(term)
                # evaluate r (bestowal)
                bestowal = self.bestow(
                    state_prime, step, action, close, trades, term)
                if i == epochs - 1:
                    rewards.append(bestowal)
                # Set up memory for reinforcement learning
                if len(st_mem) < buffer:
                    st_mem.append([state, action, bestowal, state_prime])
                # if memory is full, overwrite values
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
                        # if term:

                        # print(mem_state_prime.shape)
                        max_mem_q = np.max(
                            self.rnn.predict_on_batch(mem_state_prime), axis=2)
                        if term:
                            update = [mem_bestowal]
                        else:
                            update = mem_bestowal + (gamma * max_mem_q)
                        y = mem_q
                        for k in range(len(self.symbols)):
                            # ipdb.set_trace()

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

            if ((i + 1) % 20 == 0) or (i == 0):
                epoch_bestowal = self.eval_Q(self.data_test, ep=i + 1)
                self.learning_progress.append((i + 1, epoch_bestowal))
            else:
                epoch_bestowal = None
                # print(epoch_bestowal[-1])
            # So we know what's going on.
            # import ipdb; ipdb.set_trace()
            print('\n\nEPOCH: {} \nREWARD: {}\nEXPLORATION COEFFICENT: {}\n\n'.format(
                i + 1, epoch_bestowal, explore))
            # message = client.messages.create(to=to_num, from_=from_num, body = '\n\nEPOCH {} FINISHED\nREWARD: {}\nEXPLORATION COEFFICENT: {}\n\n'.format(i+1, epoch_bestowal[-1][-1], explore))
            if explore > 0.1:
                explore -= (3 / epochs)
            else:
                explore = .01
            # if i % 5 == 4:

            if (i % 10 == 0):
                self.rnn.save('./data/model{}sgd.h5'.format(i), overwrite=True)
                #     self.test(ep = i)
            # serialize model to JSON
            # model_json = self.rnn.to_json()
            # with open("./data/model{}.json".format(i), "w") as json_file:
            #     json_file.write(model_json)
            #     # serialize weights to HDF5
            #     self.rnn.save_weights("./data/model{}.h5".format(i))
            #     print("Saved model to disk")
            # if (i +1 == epochs):
            #     go = input('Continue? y/[n]')
            #     if go == 'y':
            #         i = epochs - raw_input('How many episodes?')
        eval_time = time.time() - t0

        print('Reinforcement Learning Completed in {} seconds'.format(
            round(eval_time, 2)))
        print(' -- {} seconds per Epoch\n'.format(round(eval_time / epochs, 2)))
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
        # self.load_data(np.inf, is_pkld = True, test = True)
        state, closeT = self.init_state(self.X_t, test=True)
        trades = np.array([np.zeros(len(self.symbols))] * len(self.data_test))
        # step = 14
        # state = self.data_test[step:step+1, :,:]
        values = []
        actions = []
        term = False
        scaler = joblib.load('./data/close_standardScaler.pkl')
        step = 1
        while not term:
            st = scaler.transform(state[0, :, :])[np.newaxis, :, :]
            qs = self.rnn.predict_on_batch(st)
            # if abs(qs[:,:,1] - qs[:,:,2]) < abs(qs[:,:,2]*0.1):
            #     action = np.zeros(len(self.symbols)).astype(int)
            # else:
            #     # import ipdb; ipdb.set_trace()
            #     action = [np.zeros(len(self.symbols)).astype(int),np.argmax(qs, axis = 2)[0,:]][np.random.choice(2)]
            action = np.argmax(qs, axis=2)[0]
            values.append(qs)
            print(qs)
            actions.append(action)
            # print(action)
            if step + 1 == len(self.data_test):
                term = True
            state_prime, step, trades, term = self.act(
                self.data_test, state, action, trades, step)
            bestowal = self.bestow(
                state_prime, step, action, closeT, trades, term=True, evaluate=True)
            # print(bestowal)
            state = state_prime
            if term:
                # ipdb.set_trace()
                # print(qs)

                # for i, s in enumerate(self.symbols):

                    # scaler = joblib.load('./data/{}_close_standardScaler.pkl'.format(s))
                    # close = scaler.inverse_transform(closeT)
                    # b_test = bt.Backtest(pd.Series(data=close[i,:], index = np.arange(len(trades))), pd.Series(data = [j[i] for j in trades]) , initialCash = self.init_cash, roundShares = False, signalType='shares')
                    # b_test.data['delta'] = b_test.data['shares'].diff().fillna(0)
                    # plt.figure(figsize = (12,8))
                    # plt.subplot(2,1,1)
                    # b_test.plotTrades()
                    # plt.axvline(x=round(self.data_size*0.8), color='black', linestyle='--')
                    # # plt.text(250, 400, 'training data')
                    # # plt.text(450, 400, 'test data')
                    # plt.subplot(2,1,2)
                    # b_test.pnl.plot()
                    # plt.suptitle(self.symbols[i] + str(ep))
                    # plt.savefig('./images/{}_{}TEST_summary.png'.format(self.symbols[i], ep),bbox_inches = 'tight',pad_inches = 0.5, dpi = 60)
                    # plt.close('all')
                    # self.init_cash = 1000
                    # joblib.dump(b_test.data, './data/btest_test_{}{}.pkl'.format(s,i))
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
        message = client.messages.create(
            to=to_num, from_=from_num, body="COMPUTE FINISHED -- SUCCESS")
        # print(message.sid)
    except:

        # rewards, Qs, actions, closeT ,b_test= model.test(ep = 100)
        # message = client.messages.create(to=to_num, from_ = from_num, body = "COMPUTE FAILED \n Unexpected Error: {}".format(str(sys.exc_info()[0]).split()[1].strip(punctuation)))
        # print(message.sid)
        raise
