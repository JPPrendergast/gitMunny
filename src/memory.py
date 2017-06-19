import numpy as np

class Memory(object):
    '''
    Base class
    '''
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def remember(self, state, action, bestowal, state_prime, term):
        raise NotImplementedError

    def get_batch(batch_size =1):
        raise NotImplementedError


class ExperienceReplay(Memory):
    '''
    Experience Replay Table

    parameters
    ----------
    memory_length : int
        How many memories are held

    Attributes
    ----------
    memory : list
        list with elements [state, action, bestowal, state_prime, term]
    experience : list
        Same as 'memory' but with pre-loaded memories from previous experiment.
        Not changed during Training
    remember : callable
        stores a new set of elements (same as memory) to memory
    get_batch : callable
        sample elements form memory and experience and calculates input and desired outputs for training a model
    '''
    def __init__(self, memory_length = 1, experience = None):
        self.memory_length = memory_length
        self.experience = experience
        self.memory = list()

    def reset(self, experience = None):
        if experience:
            self.experience = experience
        self.memory = list()

    def remember(self, state, action, bestowal, state_prime, term):
        self.memory.append({
            'state' : state,
            'action': action,
            'bestowal': bestowal,
            'state_prime': state_prime,
            'term' : term
        })
        if len(self.memory) > self.memory_length:
            del self.memory[0]

    def get_batch(self, model, batch_size = 1, exp_batch_size = 0, gamma = 0.95, callback = None):
        '''
        Get Batch
        get input, target samples from memory

        parameters
        ----------
        model: Model object
            The RNN model to be trained
        batch_size : int
            Number of samples form :attr: 'ExperienceReplay.memory'
        exp_batch_size : int
            Number of samples from :attr: 'ExperienceReplay.experience'
        gamma : float
            \gamma discount factor of future rewards
        callback : callable
            A callback to calculate target values from Model object and 'state_prime'. if None, use a default Q-learning target:
            t = r_t + \gamma argmax_{a'} Q(s', a')

        Returns
        -------
        inputs : 'numpy.array'
            Input observations of 'current' states
        targets : 'numpy.array'
            Target values of the 'future' states
        '''
        batch_mem = min(batch_size, len(self.memory))
        if exp_batch_size > 0:
            batch_exp = min(exp_batch_size, len(self.experience))
        else:
            batch_exp = 0
        bsize = batch_mem + batch_exp
        inputs = np.zeros((bsize,) + model.input_shape)
        actions = np.zeros((bsize,1))
        targets = np.zeros((bsize, 1))

        #sample from memory

        h = np.random.randint(0, len(self.memory)-batch_mem)
        rlst = np.arange(h, h+batch_mem)
        for i, m in enumerate(rlst):
            state = self.memory[m]['state']
            bestowal = self.memory[m]['bestowal']
            state_prime = self.memory[m]['state_prime']
            inputs[i] = state.reshape((1,) + model.input_shape)
            actions[i] = self.memory[m]['action']
            state_prime = state_prime.reshape((1,) + model.input_shape)
            if callback:
                targets[i] = callback(model, state_prime)
            else:
                targets[i] = bestowal + (gamma * model.max_values(state_prime, train = True))

        #sample from experience

        if not self.experience and exp_batch_size > 0:
            return inputs, targets, actions
        else:
            h = np.random.randint(0, len(self.memory)-batch_exp)
            rlst = np.arange(h, h+batch_exp)
            for k, mem in enumerate(rlst):
                state = self.memory[mem]['state']
                bestowal = self.memory[mem]['bestowal']
                state_prime = self.memory[mem]['state_prime']
                inputs[i+k] = state.reshape((1,) + model.input_shape)
                actions[i+k]= self.memory[mem]['action']
                state_prime = state_prime.reshape((1,) + model.input_shape)
                if callback:
                    targets[i+k] = callback(model, state_prime)
                else:
                    targets[i+k] = bestowal + (gamma * model.max_values(state_prime, train = False))
            return inputs, targets, actions

    @property
    def description(self):
        dstr = 'experience Replay \n\t Memory Length: {}'
        return dstr.format(self.memory_length)
