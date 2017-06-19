import numpy as np
from tradingWithPython import backtest


class Environment(object):
    '''
    Base Environment Class

    Attributes
    ----------
    observe : callable
        get observation of the environments state
        In this case, cryptomarket price analytics
    update : callable
        Update environments state from a given action
    reset : callable
        reset Environment to an initial state
    bestow : callable
        Bequeath upon thine agent the consequenses of its actions
    term : bool
        terminal state (if the episode is finished)
    state : list
        Time-series representation of the environments state
    '''
    def __init__(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def bestow(self):
        raise NotImplementedError

    @property
    def term(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    @property
    def description(self):
        raise NotImplementedError

class Trader(Environment):
    '''
    Trader

    Agent is a cryptocurrency trader. Reward is equivalent to the value gained/
        lost by the agent over the course of the episode.

    Attributes
    ----------
    num_prices : int
        Numer of prices to use per episode
    output_type : str
        'full' : Use full technical analysis of price data as a feature in state
            State space size : 9
        'close': Only use 'close' prices as feature
            State space size : 3
    '''
    def __init__(self, num_prices=84, output_type='full', init_cash = 1000, init_coins = 2, sell_coins = 0.1):
        self.n_prices = num_prices
        self.output_type = output_type
        self.init_cash = init_cash
        self.init_coins = init_coins
        self.sell_coins = sell_coins

    def _init_state(self, data, train = True):
        self._state = data[0,:]

    def _update_state(self, indata, step, action):
        state = self._state
        mask = [0,-self.sell_coins,self.sell_coins, 0]
        state = indata[step: step + 1,:]
        
















'''
placeholder
'''
