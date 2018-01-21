import argparse

# Parameters

parser = argparse.ArgumentParser(description='gitMunny')
parser.add_argument('--period', dest='period', type=int,
                    default=14400, help='Seconds between price metrics.')
parser.add_argument('--start', dest='start', type=int,
                    default=1439013600, help='UNIX timestamp -- start date.')
parser.add_argument('--end', dest='end', type=int,
                    default=1497916800, help='UNIX timestamp -- end date.')
parser.add_argument('--epochs', dest='epochs', type=int,
                    default=1000, help='Number of epochs.')
parser.add_argument('--batch', dest='batch', type=int, default=8,
                    help='Batch size retrieved from experience replay memory.')
parser.add_argument('--epsilon', dest='epsilon', type=float,
                    default=0.1, help='Exploration Rate.')
parser.add_argument('--gamma', dest='gamma', type=float,
                    default=0.95, help='Discount rate.')
parser.add_argument('--save', dest='save', type=str,
                    default='trading_agent.h5', help='path to save agent.')
parser.add_argument('--nprices', dest='nprices', type=int,
                    default=84, help='path to save agent.')
parser.add_argument('--memory', dest='memory', type=int,
                    default=14, help='Experience replay memory length.')
args = parser.parse_args()

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, Nadam, SGD


from environment import Trader
from agent import DiscreteAgent
from market import BitcoinData
from memory import ExperienceReplay
from policies import Max
from models import KerasModel

BTCdata = BitcoinData(start=args.start, end=args.end, period=args.period)

model = Sequential()
model.add(LSTM(32,
               input_shape=(1, 7),
               return_sequences=True,
               stateful=False, activation='linear'))
model.add(Dropout(0.3))

model.add(LSTM(32,
               return_sequences=True,
               stateful=False, activation='linear'))
model.add(Dropout(0.3))

model.add(Dense(4, kernel_initializer='lecun_uniform', activation='linear'))

keras_model = KerasModel(model)

M = ExperienceReplay(memory_length=args.memory)

A = DiscreteAgent(keras_model, M)
A.compile(optimizer=SGD(lr=0.2), loss='mse', policy_rule='max')
trader = Trader(mkt=BTCdata, num_prices=args.nprices)
A.learn(trader, epochs=args.epochs, batch_size=args.batch)

A.trade(trader, epochs=1)
