import numpy as np

class ReacherAgent():

    def __init__(self, brain_name, agent_index, action_size,
                 action_min=-1, action_max=1):

        self.brain_name = brain_name
        self.agent_index = agent_index

        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max

    def act(self, state):
        action = np.random.randn(self.action_size)
        action = np.clip(action, a_min=self.action_min, a_max=self.action_max)

        return action
