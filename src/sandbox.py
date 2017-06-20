import argparse

# Parameters

parser = argparse.ArgumentParser(description='gitMunny')
parser.add_argument('--period', dest='period',type=int,default=14400,help='Seconds between price metrics.')
parser.add_argument('--start', dest='start',type=int,default=1439013600,help='UNIX timestamp -- start date.')
parser.add_argument('--end', dest='end',type=int,default=1497916800,help='UNIX timestamp -- end date.')
parser.add_argument('--epochs', dest='epochs',type=int,default=1000,help='Number of epochs.')
parser.add_argument('--batch', dest='batch',type=int,default=8,help='Batch size retrieved from experience replay memory.')
parser.add_argument('--epsilon', dest='epsilon',type=float,default=0.1,help='Exploration Rate.')
parser.add_argument('--gamma', dest='gamma',type=float,default=0.95,help='Discount rate.')
parser.add_argument('--save', dest='save',type=str,default='trading_agent.h5',help='path to save agent.')






from environment import Trader
from agent import DiscreteAgent
from market import BitcoinData
from memory import ExperienceReplay
from policies import Max
from model import KerasModel
