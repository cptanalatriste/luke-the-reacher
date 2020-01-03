from collections import namedtuple, deque
import random
import torch
import copy


import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process. Code taken from:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
    """

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        random_list = [random.random() for i in range(len(x))]
        dx = self.theta * (self.mu - x) + self.sigma * np.array(random_list)
        self.state = x + dx
        return self.state

class ReplayBuffer():
    """
    A repository of agent experiences. While training, experiences will be
    provided from here.

    Code taken from: https://github.com/cptanalatriste/banana-hunter
    """

    def __init__(self, buffer_size, training_batch_size, device):

        self.storage = deque(maxlen=buffer_size)
        self.training_batch_size = training_batch_size
        self.device = device

        self.field_names = ['state', 'action', 'reward', 'next_state', 'done']
        self.field_types = [np.float32, np.int64, np.float32, np.float32, np.float32]

        self.experience = namedtuple("Experience", field_names=self.field_names)

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the repository.
        """
        self.storage.append(self.experience(state, action, reward,
                                            next_state, done))

    def sample(self):
        """
        Sample a set of experiences from the repository.
        """
        raw_samples = random.sample(self.storage, k=self.training_batch_size)
        values = []

        for tuple_index in range(len(self.field_names)):
            value_list = [sample[tuple_index]
                          for sample in raw_samples if sample is not None]

            value_list = np.vstack(value_list).astype(self.field_types[tuple_index])
            value_list = torch.from_numpy(value_list)
            values.append(value_list.to(self.device))

        return tuple(values)

    def __len__(self):
        return len(self.storage)
