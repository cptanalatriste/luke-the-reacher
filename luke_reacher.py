import numpy as np

class ReacherAgent():

    def __init__(self, action_size, action_min=-1, action_max=1):

        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max

    def act(self, state):
        action = np.random.randn(self.action_size)
        action = np.clip(action, a_min=self.action_min, a_max=self.action_max)

        return action

    def reset(self):
        pass

    def save_trained_weights(self, network_file):
        pass
