from __future__ import absolute_import
import numpy as np
import keras.backend as K
from keras import optimizers, objectives, regularizers
from keras.engine.training import _weighted_masked_objective, losses
import policies

import itertools


class Model(object):
    '''
    Base Model class
    Attributes
    ----------
    compile : callable
        Initialize model weights or values
    values : callable
        return action values given input observation
    max_values : callable
        np.max of `values`
    policy : callable
        Return action given an input enviroment observation as np.argmax of
        `values`
    update : callable
        Update the model in response to state transitions.
    num_action : int
        Number of possible actions
    input_shape : :list:`int`
        List with dimensions of the input observation
    '''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def compile(self):
        raise NotImplementedError

    def values(self, observation, train=False):
        raise NotImplementedError

    def max_values(self, observation, train=False):
        raise NotImplementedError

    def policy(self, observation, train=False):
        raise NotImplementedError

    def update(self, inputs, targets, actions):
        raise NotImplementedError

    @property
    def num_actions(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def description(self):
        raise NotImplementedError


class KerasModel(Model):
    '''
    Keras model wrapper

    Attributes
    ----------
    model: Sequential Keras Model
    '''

    def __init__(self, keras_model):
        self.keras_model = keras_model

    def compile(self, optimizer='sgd', loss='mse', policy_rule='max', sample_weight_mode=None):
        '''
        Initialixe model wieghts and compile functions
        '''
        kmodel = self.keras_model
        kmodel.build()
        self.policy_rule = policies.Max()
        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode

        self.loss = losses.get(loss)
        weighted_loss = _weighted_masked_objective(self.loss)

        self.X_train = kmodel.get_input_at(0)
        self.X_test = kmodel.get_input_at(0)

        values_train = kmodel.get_output_at(0)
        values_test = kmodel.get_output_at(0)
        self.y_train = self.policy_rule(values_train)
        self.y_test = self.policy_rule(values_test)
        self.y = K.placeholder(ndim=K.ndim(self.y_train))

        if self.sample_weight_mode == 'temporal':
            self.weights = K.placeholder(ndim=3)
        else:
            self.weights = K.placeholder(ndim=2)

        if hasattr(kmodel.layers[-1], 'get_output_mask'):
            mask = kmodel.layers[-1].get_output_mask()
        else:
            mask = None
        train_loss = weighted_loss(self.y, self.y_train, self.weights, mask)
        test_loss = weighted_loss(self.y, self.y_test, self.weights, mask)
        import ipdb
        ipdb.set_trace()
        train_loss = regularizers.l2(train_loss)

        updates = self.optimizer.get_updates(
            kmodel.trainable_weights, kmodel.constraints, train_loss)
        updates += kmodel.updates

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.y, self.weights]
            test_ins = self.X_test + [self.y, self.weights]
            assert type(self.X_test) == list
            values_ins_test = self.X_test
            values_ins_train = self.X_train
        else:
            train_ins = [self.X_train, self.y, self.weights]
            test_ins = [self.X_test, self.y, self.weights]
            values_ins_test = [self.X_test]
            values_ins_train = [self.X_train]

        self._train = K.function(train_ins, [train_loss], updates=updates)
        self._values_train = K.function(
            values_ins_train, [values_train], updates=kmodel.state_updates)
        self._values_test = K.function(
            values_ins_test, [values_test], updates=kmodel.state_updates)

        self._test = K.function(
            test_ins, [test_loss], updates - kmodel.state_updates)

    def values(self, observation, train=False):
        if train:
            vals = self._values_train([observation])[0]
        else:
            vals = self._values_test([observation])[0]
        return vals

    def max_values(self, observation, train=False):
        vals - self.values(observation, train)
        return self.policy_rule.max(vals)

    def policy(self, observation, train=False):
        vals = self.values(observation, train)[0]
        return self.policy_rule.policy(vals)

    def update(self, inputs, targets, actions, weights=None):
        if weights is None:
            weights = np.ones(len(targets))
        loss = self._train([inputs, targets, weights])[0]
        return loss

    @property
    def description(self):
        dstr = 'Keras Model \n\t Optimixer {} \n\t Loss: {} \n\t Policy: {}'
        return dstr.format(self.optimizer, self.loss, self.policy_rule)


class TableModel(Model):
    def __init__(self, state_dim, num_actions):
        '''
        Table model
        stores and updates a Q-value for every possible state/action pair

        Attributes:
        -----------
        state_dim : dimension of state space
        num_actions: size of action space
        '''

        self.state_dim = state_dim
        self.n_actions - num_actions

    def compile(self, state_dim_values, lr=0.2, policy_rule='maxrand', init_value=None):
        '''
        build and initialize table with all possible state values.
        state_dim_values consists of a tuple of arrays or lists.
        '''

        self.policy_rule = policies.get(policy_rule)

        if init_value is None:
            self.init_value = np.zeros(self.num_actions)
        else:
            self.init_value = init_value

        self.table = {key: np.array(self.init_value) for key in list(
            itertools.product(*state_dim_values))}
        self.lr = lr

    def values(self, observation):
        if observation.ndim == 1:
            vals = self.table[tuple(observation)]
        else:
            obs_tuple = tuple(map(tuple, observation))
            vals = map(self.table.__getitem__, obs_tuple)
        vals = np.asarray(vals)
        return vals

    def max_values(self, observation, *args, **kwargs):
        vals = self.values(observation)
        return self.policy_rule.max(vals)

    def policy(self, observation, *args, **kwargs):
        vals = self.values(observation)
        return self.policy_rule.policy(vals)

    def update(self, inputs, targets, actions, weights=None):
        current_values = self.values(inputs)

        for i, input in enumerate(inputs):
            a = int(actions[i])
            self.table[tuple(input)][a] = current_values[i][a] + \
                self.lr * (targets[i] - current_values[i][a])

        return((targets - current_values)**2).sum()

    @property
    def num_actions(self):
        return self.n_actions

    @property
    def input_shape(self):
        return (self.state_dim,)

    @property
    def description(self):
        dstr = 'Table Model'
        return dstr
