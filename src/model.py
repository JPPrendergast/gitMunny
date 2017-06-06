from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam, Nadam
from sklearn.preprocessing import StandardScaler
import talib.abstract as ta
import backtest as twp
import pandas as pd
import numpy as np

import time

class Model(object):
    def __init__(self,tsteps = 1, batch_size = 1, num_features = 7, drop = 0.33):

        model = Sequential()
        model.add(LSTM(64,
                       input_shape = (1, num_features),
                       return_sequences = True,
                       stateful = False))
        model.add(Dropout(dropout))
        model.add(LSTM(64,
                       input_shape = (1,num_features),
                       return_sequences = False,
                       stateful = False))
        model.add(Dropout(dropout))
        model.add(dense(4, init = 'lecun_uniform'))
        model.add(Activation('relu'))

        model.compile(loss = 'rmse', optimizer = adam)
        self.model = model
        
    def load_data(filename, quantity, test = False):
        prices = pd.read_csv(filename)
        X_train = prices.iloc[-quantity:-round(quantity*0.2),]
        X_test = prices.iloc[-quantity:,]
        if test:
            return X_test
        else:
            return X_train

    def init_state(indata, test=False):
        close = indata['close'].values
        diff = np.diff(close)
        diff = np.insert(diff, 0,0)
        sma15 = ta.SMA(indata, timeperiod = 15)
        sma60 = ta.SMA(indata, timeperiod = 60)
        rsi = ta.RSI(indata, timeperiod = 14)
        atr = ta.ATR(indata, timeperiod = 14)

        x_data = np.column_stack((close, diff, sma15, close-sma15, sma15-sma30, rsi, atr))
        x_data = np.nan_to_num(x_data)
        if test:
            scaler = joblib.load('data/scaler.pkl')
            x_data = np.expand_dims(scaler.transform(x_data), axis = 1)
        else:
            scaler = StandardScaler()
            x_data = np.expand_dims(scaler.fit_transform(x_data), axis = 1)
        state = x_data[0:1, 0:1, :]

        self.state, self.x_data, self.close = state, x_data, close
        return self

    def take_action(state, x_data, action, signal, time_step):
        time_step += 1
        if time_step + 1 == x_data.shape[0]:
            state = x_data[time_step-1:time_step, 0:1, :]
            terminal_state = 1
            signal.loc[time_step] = 0
            return state, time_step, signal, terminal_state

        state = x_data[time_step-1: time_step, 0:1, :]
        if action == 1:
            signal.loc[time_step] = 100
        elif action == 2:
            signal.loc[time_step] = -100
        else:
            signal.loc[time_step] = 0
        terminal_state = 0

        self.state, self.time_step, self.signal, self.terminal_state = state, time_step, signal, terminal_state

    def get_reward(new_state, time_step, action, x_data, signal, terminal_state, evaluate = False, epoch = 0):
        reward = 0
        signal.fillna(value = 0, inplace = True)

        if not evaluate:
            bt = twp.Backtest(pd.Series(data=[x for x in x_data[time_step-2:time_step]], index = signal[time_step-2:time_step].index.values),signal[time_step-2: time_step], signalType = 'shares')
            reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2])*bt.data['shares'].iloc[-1])

        if terminal_state == 1 and evaluate:
            bt = twp.Backtest(pd.Series(data=[x for x in xdata], index=signal.index.values), signal, signalType='shares')
            reward = bt.pnl.iloc[-1]
            plt.figure(figsize=(3,4))
            bt.plotTrades()
            plt.axvline(x=400, color='black', linestyle='--')
            plt.text(250, 400, 'training data')
            plt.text(450, 400, 'test data')
            plt.suptitle(str(epoch))
            plt.savefig('plt/'+str(epoch)+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
            plt.close('all')

        self.reward = reward
        return self

    def evaluate_Q(eval_data, eval_model, price_data, epoch = 0):
        signal = pd.Series(index = np.arange(len(eval_data)))
        state, x_data, price_data = init_state(eval_data)
        status = 1
        terminal_state = 0
        time_step = 1
        while status == 1:
            qval = eval_model.preict(state, batch_size = 1)
            action = (np.argmax(qval))
            new_state, time_step, signal, terminal_state = take_action(state, x_data, action, signal, time_step)
            eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, evaluate = True, epoch = epoch)
            state = new_state
            if terminal_state == 1:
                status = 0
        self.eval_reward = eval_reward
        return self

    def fit(data, epochs = 100):
