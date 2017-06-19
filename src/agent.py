import numpy as np

from keras.utils.generic_utils import Progbar

class Agent(object):
    '''
    Base Agent Class

    Parameters
    ----------
    model: Model object
        In this case, the RNN instantiated in ~src/model.py
    memory: Memory object
    '''
    def __init__(self, model, memory):
        self.model = model
        self.memory = memory


class DiscreteAgent(Agent):
    '''
    Single discrete action agent. (i.e. it makes one well-defined
        decision per episode)

    Parameters
    ----------
    model: Model object
    memory: Memory object
        Models memory for storing experiences for replay
    epsilon : callable
        a rule to define if model will explore or exploit
    '''
    def __init__(self, model, memory, epsilon = None):
        super(DiscreteAgent, self).__init__(model,memory)
        if epsilon is None:
            epsilon = lambda *args: .1
        self.epsilon = epsilon

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)
        if 'experience' in kwargs:
            experience = kwargs['experience']
        else:
            experience = None
        self.memory.reset(experience)

    def values(self, observation, train = False):
        return self.model.values(observation, train)

    def max_values(self, observation, train = False):
        return self.model.max_values(observation, train)

    def policy(self, observation, train = False):
        if train and np.random.rand() <= self.epsilon():
            return[np.random.randint(o, self.num_actions)]
        else:
            return self.model.policy(observation, train)

    def update(self, batch_size = 1, exp_batch_size = 0, gamma = 0.95, callback = None):
        inputs, targets, actions = self.get_batch(self.model, batch_size = batch_size, exp_batch_size = exp_batch_size, gamma = gamma, callback = callback)
        loss = self.model.update(inputs, targets, actions)
        return loss

    @property
    def num_actions(self):
        return self.model.num_actions

    @property
    def input_shape(self):
        return self.model.input_shape

    def reset(self):
        self.memory.reset()

    def remember(self, state, action, bestowal, state_prime, term):
        self.memory.remember(state, action, bestowal, state_prime, term)

    def get_batch(self, model, batch_size = 1, exp_batch_size = 0, gamma = 0.95, callback = None):
        return self.memory.get_batch(model, batch_size, exp_batch_size, gamma, callback)

    def learn(self, env, epochs=1, batch_size=1, exp_batch_size=0, gamma = 0.95, reset_memory = False, verbose = 1, callbacks = None):
        '''
        Train agent (RNN) to interact with the environment

        Parameters
        ----------
        env : Environment object
        epochs : int
            number of episodes
        batch_size : int
            number of experiences to replay per step
        exp_batch_size : int
            number of experiences to replay from the consolidated
            :attr: 'ExperienceReplay.experience'
        gamma : float
            discount factor
        reset_memory : bool
            True for a "cold-start"
            -- erases :attr: `ExperienceReplay.memory`
        verbose : int
            This one's obvious
        callbacks : list of callables
        '''
        print('Beginning gitMunny Learning\n')
        print("[Environment]: {}".format(env.description))
        print("[Model]: {}".format(model.description))
        print("[Memory]: {}".format(memory.description))

        if reset_memory:
            self.reset()
        progbar = Progbar(epochs)

        for i in range(epochs):
            
