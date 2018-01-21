from __future__ import absolute_import
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

    def update(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reward(self):
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

    def __init__(self, mkt, num_prices=84, init_cash=1000, init_coins=2, sell_coins=0.1):
        self.n_prices = num_prices
        self.output_type = output_type
        self.init_cash = init_cash
        self.init_coins = init_coins
        self.sell_coins = sell_coins
        # self.data = data
        self.mkt = mkt

    def reset(self, test=False):
        if not test:
            n = self.n_prices
            self.data = mkt.scaler.transform(mkt.random_train_data(n))
        else:
            self.data = mkt.scaler.transform(mkt.test_data)
            self.n_prices = len(self.data)
        self._state = np.array(
            self.data[0, :], self.init_coins * self.data[0, 0], self.initialCash)
        self._state_prime = self.data[1, :]
        self.step = 0

    def update(self, action):
        self._update_state(action)
        reward = self.reward()
        self._state = self._state_prime
        self.step += 1
        term = self.term
        if not term:
            self._state_prime = self.data[step + 1]
        return self.sp, reward, term

    def _update_state(self, action):
        state = self._state
        state_prime = self._state_prime
        mask = np.array([0, -self.sell_coins, self.sell_coins, 0])
        term = False
        new_val = max(0, self.sp[-2] + (mask[action] * state[0]))
        if new_val != 0:
            new_cash = max(0, self.sp[-1] - (mask[action] * state[0]))
        else:
            new_cash = self.sp[-1]

        if new_cash == 0:
            new_val = self.sp[-2]

        state_prime.extend(new_val, new_cash)
        self._state_prime = state_prime

    def reward(self):
        return self._state_prime[6:].sum() - self._state[6:].sum()

    @property
    def observe(self):
        return self._state

    @property
    def sp(self):
        return self._state_prime

    @property
    def term(self):
        if self.step == self.num_prices:
            return True
        else:
            return False

    @property
    def description(self):
        return "Trading Environment"


'''
placeholder
'''
