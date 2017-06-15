
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

# Machine Learning Libs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, Nadam
from string import punctuation
#Trading libraries
import talib.abstract as ta
from poloniex import Poloniex
# from tradingWithPython
import backtest as bt
import progressbar
import ipdb

# Globals
account = os.environ['TWILIO_API_KEY']
twi_auth = os.environ['TWILIO_AUTH_TOKEN']
to_num = os.environ['JOHN_NUM']
from_num = os.environ['FROM_NUM']
client = Client(account, twi_auth)
chump = []

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
        model.add(LSTM(128,
                       input_shape=(num_curr, num_features),
                       return_sequences=True,
                       stateful=False))
        model.add(Dropout(drop[0]))

        model.add(LSTM(128,
                       return_sequences=True,
                       stateful=False))
        model.add(Dropout(drop[1]))
        # model.add(LSTM(64,
        #                return_sequences = True,
        #                stateful = False))
        # model.add(Dropout(drop[2]))
        model.add(Dense(4, kernel_initializer='lecun_uniform', activation = 'linear'))
        # model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        model.compile(loss='mse', optimizer='rmsprop', sample_weight_mode = 'temporal')
        self.rnn = model
        self.symbols = symbols
        self.batch_length = batch_length
        self.init_cash = 1000

    def load_data(self, quantity, start = None, end = None, period = 14400, is_pkld = True, test = False):
        '''
        5 largest Cryptos by market volume = ['BTC', 'ETH', 'XRP', 'ETC', 'XMR']

        INSERT DOCSTRING HERE
        '''
        polo_api = os.environ['POLO_API']
        polo_secret = os.environ['POLO_SECRET_KEY']
        if is_pkld:
            prices = joblib.load('../data/prices_{}coins.pkl'.format(len(self.symbols)))
        else:
            sym_price = []
            a = []
            b = []
            for i in self.symbols:
                c = pd.DataFrame.from_records(Poloniex(key = polo_api, secret = polo_secret).returnChartData(currencyPair = 'USDT_' + i, start = start, end = end, period = period), index = 'date')
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
        self.data_size = min(quantity, len(prices))
        self.X_train, self.X_test = self.split_data(prices, self.data_size, 0.2)
        if not test:
            self.X_t = self.X_train.append(self.X_test)
            # _,_ = self.init_state(self.X_train)
            # _,_ = self.init_state(self.X_test, )
        if test:
            self.X_t = prices
        return self

    def split_data(self, data, quantity, test_size):
        '''
        INSERT DOCSTRING HERE
        '''
        if quantity < len(data):
            l = len(data) - quantity
            r = np.random.randint(l)
        else:
            quantity = len(data)-1
            r = 0
        # self.X_test_little = data.iloc(-((len(data)-quantity)-round(quantity*test_size)):)
        return data.iloc[r:int(quantity*(1-test_size))+r,], data.iloc[r+int(quantity*(1-test_size)):r+quantity]

    def init_state(self, X, test):
        '''
        INSERT DOCSTRING HERE
        '''
        data  = []
        close = []
        #print(self.symbols)
        for i, s in enumerate(self.symbols):
            # import ipdb; ipdb.set_trace()
            if not test:
                scaler = StandardScaler()
                close.append(scaler.fit_transform(X[s]['close'].values.reshape(-1,1)).flatten())
                joblib.dump(scaler, '../data/{}scaler.pkl'.format(s))
            elif test:
                scaler = joblib.load('../data/{}scaler.pkl'.format(s))
                close.append(scaler.transform(X[s]['close'].values.reshape(-1,1)).flatten())
            diff = np.diff(close[i])
            diff = np.insert(diff, 0,0)
            sma15 = ta.SMA(X[s], timeperiod = 15)
            sma60 = ta.SMA(X[s], timeperiod = 60)
            rsi = ta.RSI(X[s], timeperiod = 14)
            atr = ta.ATR(X[s], timeperiod = 14)

            data.append(np.nan_to_num(np.vstack((close[i], diff, sma15, close[i]-sma15, sma15-sma60, rsi, atr))))
            data[i] = np.expand_dims(data[i], axis = 1)

        data = np.hstack(data).T
        close = np.vstack(close)
        state = data[0:1,:,:]
        if test:
            self.data_test = data
        else:
            self.data = data
        return state, close


    def act(self,indata, state, action, trades, step):
        '''
        INSERT DOCSTRING HERE
        '''
        # print(action)
        mask = np.array([0,10,-10,0])
        # print(step, indata.shape[0])
        if step +1 == indata.shape[0]:
            state = indata[step, :, :]
            state = np.expand_dims(state, axis = 0)
            term = True
            trades[step] = [0]*len(self.symbols)
            return state, step, trades, term
        else:
            state = indata[step+1, :, :]
            # Take trade action {0:Hold, 1:Buy, 2:Sell}
            # print(trades)
            # for i in range(len(self.symbols)):
            term = False
            # import ipdb; ipdb.set_trace()
            trades[step+1] = mask[action]
            step += 1
            # print(step)
            return state[np.newaxis, :], step, trades, term

    def bestow(self, new_state, step, action, prices, trades, term, epoch = None , plot = False, evaluate = False):
        '''
        INSERT DOCSTRING HERE
        '''
        bestowal = [0]*len(self.symbols)
        if evaluate == False:
            for i in range(len(self.symbols)):
                # ipdb.set_trace()
                b_test = bt.Backtest(pd.Series(data=prices[i,step-2:step], index =np.arange(step-2,step)).astype(float),pd.Series(data = trades[step-2:step,i], index = np.arange(step-2,step)).astype(float),roundShares = False, signalType='capital')
                bestowal[i] = ((b_test.data['price'].iloc[-1] -b_test.data['price'].iloc[-2]) * b_test.data['trades'].iloc[-1])

        elif evaluate == True and term == True:
            for i in range(len(self.symbols)):
                # ipdb.set_trace()
                scaler = joblib.load('../data/{}scaler.pkl'.format(self.symbols[i]))
                close = scaler.inverse_transform(prices[i,:])
                b_test = bt.Backtest(pd.Series(data=prices[i,:], index = np.arange(len(trades))), pd.Series(data = trades[:,i]) ,roundShares = False, initialCash = self.init_cash, signalType='capital')
                # bestowal[i] = ((b_test.data['price'].iloc[-1] -b_test.data['price'].iloc[-2]) * b_test.data['shares'].iloc[-2])

                bestowal[i] = b_test.pnl.iloc[-1]
                if (plot == True):
                    plt.figure(figsize = (12,8))
                    plt.subplot(2,1,1)
                    b_test.plotTrades()
                    plt.axvline(x=round(self.data_size*0.8), color='black', linestyle='--')
                    # plt.text(250, 400, 'training data')
                    # plt.text(450, 400, 'test data')
                    plt.subplot(2,1,2)
                    b_test.pnl.plot()
                    plt.suptitle(self.symbols[i] + str(epoch))
                    plt.savefig('../images/{}_{}TEST_summary.png'.format(self.symbols[i], epoch),bbox_inches = 'tight',pad_inches = 0.5, dpi = 60)
                    plt.close('all')
                    self.init_cash = 1000
                # print(trades)
        self.init_cash = 1000
            # joblib.dump(b_test.data, '../data/epoch_{}_backtest.pkl'.format(epoch))
        return bestowal

    def eval_Q(self, eval_data, ep):
        '''
        INSERT DOCSTRING HERE
        '''
        bestowal = []
        state, closeT = self.init_state(self.X_t, test = True)
        trades = np.array([np.zeros(len(self.symbols))]*len(self.data_test))
        step = 14
        values = []
        actions = []
        term = False
        # import ipdb; ipdb.set_trace()
        while not term:
            qs = self.rnn.predict(state)

            action = np.argmax(qs, axis = 2).flatten()
            # values.append(qs)
            # actions.append(action)
            if step +1 == len(self.data_test):
                term = True

            state_prime, step, trades, term = self.act(self.data_test, state, action, trades, step)
            bestowal = self.bestow(state_prime, step, action, closeT, trades, term = term,epoch = ep, plot = term, evaluate = term)
            state = state_prime
            if term:
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
        #     q_values = self.rnn.predict(state_val.reshape(1,2,7))
        #     action = np.argmax(q_values, axis = 2)
        #     state_prime, step, trades, term = self.act(self.data_test, state_val, action, trades, step)
        #
        #     bestowal = self.bestow(state_prime,step,action,prices, trades, term,evaluate = True, epoch = ep)
        #     state = state_prime
        #     j += 1
        #     # print(bestowal)
        #     if term:
        #         return np.array(bestowal)

    def fit(self, epochs = 100):
        '''
        INSERT DOCSTRING HERE
        '''

        #reinforcement learning parameters
        gamma = 0.95
        explore = 1
        mem_chunk = 7


        # "memory"
        buffer = 14
        h = 0
        st_mem = []

        learning_progress = [] #Stores state, Action, reward (bestowal :) ) and next state
        # reinforcement learning loop
        t0 = time.time()

        # bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') '])
        quantity = 56
        bar =  progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') '], max_value = epochs * (sum([quantity, quantity+(epochs*2)])/2)).start()
        for i in range(epochs):
            self.load_data(quantity, is_pkld = True, test = False)
            state,p = self.init_state(self.X_train, test = False)
            stateT,pT = self.init_state(self.X_t, test = True)
            # Set statespace for testing:
            if i == epochs - 1:
                trades = np.array([np.zeros(len(self.symbols))]*len(self.data_test))
                t_data = self.data_test
                close = pT
                state = stateT
            elif i == 0:
                trades = np.array([np.zeros(len(self.symbols))]*len(self.data))
                # trades = pd.Series(data = [[0]*(len(self.symbols))]*len(self.data), index = np.arange(len(self.data)))
                t_data = self.data
                close = p
            go = True
            term = False

            step = 14
            # state = t_data[i:i+1,:,:]
            # close = t_data
            # if i == 0:
            #     scaler = StandardScaler()
            #     qvs = self.rnn.predict(t_data)
            #     scaler.fit(qvs[:,0,:])
            print('RL Epoch {}'.format(i))
            rewards = []
            # import ipdb; ipdb.set_trace()
            while go:
                q_values = self.rnn.predict(state)
                bar+=1
                # impliment exploration parameter!
                if (np.random.rand() < explore):
                    action = np.random.randint(4, size = len(self.symbols))
                else:
                    # if abs(q_values[:,:,1] - q_values[:,:,2]) < abs(q_values[:,:,2]*0.1):
                    #     action = np.zeros(len(self.symbols)).astype(int)
                    # else:
                    #     action = [np.zeros(len(self.symbols)).astype(int),np.argmax(q_values, axis = 2)[0,:]][np.random.choice(2)]
                    action = np.argmax(q_values, axis = 2)[0]
                # take action and evaluate new state
                # print(action, np.argmax(q_values, axis = 2))
                state_prime, step, trades, term = self.act(t_data, state, action, trades, step)
                # print(term)
                # evaluate r (bestowal)

                bestowal = self.bestow(state_prime, step, action, close, trades, term)
                if i == epochs-1:
                    rewards.append(bestowal)
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
                    for z, mem in enumerate(batch):
                        mem_state, mem_action, mem_bestowal, mem_state_prime = mem
                        mem_q = self.rnn.predict(mem_state)
                        # if term:

                        # print(mem_state_prime.shape)
                        max_mem_q = np.max(self.rnn.predict(mem_state_prime), axis = 2)
                        if term:
                            update = [mem_bestowal]
                        else:
                            update = mem_bestowal + (gamma * max_mem_q)
                        y = mem_q
                        try:
                            for k in range(len(self.symbols)):
                                # ipdb.set_trace()

                                y[0][k][action[k]] = update[0][k]

                        except IndexError:
                            term = True

                        xTrain.append(mem_state)
                        yTrain.append(y)

                    xTrain = np.squeeze(np.array(xTrain), axis = (1))
                    yTrain = np.squeeze(np.array(yTrain), axis = (1))
                    t1 = time.time()
                    self.rnn.fit(xTrain,yTrain, batch_size = mem_chunk, epochs = 5, verbose = 0)
                state = state_prime
                if term:
                    go = False


            epoch_bestowal = self.eval_Q(self.X_t, ep = i+1)
            # print(epoch_bestowal[-1])
            learning_progress.append(epoch_bestowal)
            # So we know what's going on.
            # import ipdb; ipdb.set_trace()
            print('\n\nEPOCH: {} \nREWARD: {}\nEXPLORATION COEFFICENT: {}\n\n'.format(i+1, epoch_bestowal[-1], explore))
            # message = client.messages.create(to=to_num, from_=from_num, body = '\n\nEPOCH {} FINISHED\nREWARD: {}\nEXPLORATION COEFFICENT: {}\n\n'.format(i+1, epoch_bestowal[-1][-1], explore))
            if explore > 0.1:
                explore -= (1.0/epochs)
            # if i % 5 == 4:
            quantity +=2
            # if (i % 10 == 0):
            #     self.test(ep = i)
            self.rnn.save('../data/model{}.h5'.format(i), overwrite = True)
            # serialize model to JSON
            # model_json = self.rnn.to_json()
            # with open("../data/model{}.json".format(i), "w") as json_file:
            #     json_file.write(model_json)
            #     # serialize weights to HDF5
            #     self.rnn.save_weights("../data/model{}.h5".format(i))
            #     print("Saved model to disk")
        eval_time = time.time() -t0

        print('Reinforcement Learning Completed in {} seconds'.format(round(eval_time,2)))
        print(' -- {} seconds per Epoch\n'.format(round(eval_time/epochs,2)))
        plt.figure(figsize = (12,8))
        plt.plot(learning_progress, label = 'Reward per Epoch')
        plt.savefig('reward_curve.png')
        plt.close('all')




    def test(self, ep):
        #load model here
        bestowal = []
        self.load_data(np.inf, is_pkld = True, test = True)
        state, closeT = self.init_state(self.X_t, test = True)
        trades = np.array([np.zeros(len(self.symbols))]*len(self.data_test))
        step = 14
        state = self.data_test[step:step+1, :,:]
        values = []
        actions = []
        term = False
        while not term:
            qs = self.rnn.predict(state)
            # if abs(qs[:,:,1] - qs[:,:,2]) < abs(qs[:,:,2]*0.1):
            #     action = np.zeros(len(self.symbols)).astype(int)
            # else:
            #     # import ipdb; ipdb.set_trace()
            #     action = [np.zeros(len(self.symbols)).astype(int),np.argmax(qs, axis = 2)[0,:]][np.random.choice(2)]
            action = np.argmax(qs, axis = 2)[0]
            values.append(qs)
            actions.append(action)
            # print(action)
            if step +1 == len(self.data_test):
                term = True
            state_prime, step, trades, term = self.act(self.data_test, state, action, trades, step)
            bestowal = self.bestow(state_prime, step, action, closeT, trades, term = True, evaluate = True)
            # print(bestowal)
            state = state_prime
            if term:
                # ipdb.set_trace()
                # print(qs)

                for i, s in enumerate(self.symbols):
                    scaler = joblib.load('../data/{}scaler.pkl'.format(s))
                    close = scaler.inverse_transform(closeT)
                    b_test = bt.Backtest(pd.Series(data=close[i,:], index = np.arange(len(trades))), pd.Series(data = [j[i] for j in trades]) , initialCash = self.init_cash, roundShares = False, signalType='capital')
                    b_test.data['delta'] = b_test.data['shares'].diff().fillna(0)
                    plt.figure(figsize = (12,8))
                    plt.subplot(2,1,1)
                    b_test.plotTrades()
                    plt.axvline(x=round(self.data_size*0.8), color='black', linestyle='--')
                    # plt.text(250, 400, 'training data')
                    # plt.text(450, 400, 'test data')
                    plt.subplot(2,1,2)
                    b_test.pnl.plot()
                    plt.suptitle(self.symbols[i] + str(ep))
                    plt.savefig('../images/{}_{}TEST_summary.png'.format(self.symbols[i], ep),bbox_inches = 'tight',pad_inches = 0.5, dpi = 60)
                    plt.close('all')
                    self.init_cash = 1000
                    joblib.dump(b_test.data, '../data/btest_test_{}{}.pkl'.format(s,i))
                return bestowal, values, actions, closeT, b_test


if __name__ == '__main__':
    # GOOD 28 mem chunk, 56 buffer, 100 epochs, 20 RNN epochs converged aroun 50
    # Bad: 14 mem chunk, 100 buffer, 200 epochs, 5 RNN epochs total shite
    # Current 21 mem chunk 21 buffer, 200 epochs, 50 RNN epochs also converges around 50

    coins = ['BTC']
    start = mktime(datetime(2013,1,1).timetuple())
    end = mktime(datetime(2017, 5,30).timetuple())
    model = Model(num_features = 7,symbols = coins)
    model.load_data((end-start)/86400, start = start, end = end, period = 86400, test = False, is_pkld = True)
    state, close = model.init_state(model.X_train, test = False)
    client = Client(account, twi_auth)
    try:
        model.fit(epochs = 1000)
        message = client.messages.create(to=to_num, from_ = from_num, body = "COMPUTE FINISHED -- SUCCESS")
        # print(message.sid)
    except:

        # rewards, Qs, actions, closeT ,b_test= model.test(ep = 100)
        message = client.messages.create(to=to_num, from_ = from_num, body = "COMPUTE FAILED \n Unexpected Error: {}".format(str(sys.exc_info()[0]).split()[1].strip(punctuation)))
        print(message.sid)
        raise
