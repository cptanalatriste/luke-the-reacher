from collections import namedtuple, deque
import random
import torch

import numpy as np

class ReplayBuffer():
    """
    A repository of agent experiences. While training, experiences will be
    provided from here.
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
