
from rl_manager import TrainingManager

class ReacherManager(TrainingManager):

    def __init__(self, environment_params):
        self.environment_params = environment_params

    def get_action_parameters(self):
        return {'add_noise': True}

    def do_reset(self, environment, agent):

        brain_name = self.environment_params['brain_name']
        agent_index = self.environment_params['agent_index']

        environment_info = environment.reset(train_mode=True)[brain_name]

        return environment_info.vector_observations[agent_index]

    def do_step(self, environment, action):
        brain_name = self.environment_params['brain_name']
        agent_index = self.environment_params['agent_index']

        environment_info = environment.step(action)[brain_name]

        next_state = environment_info.vector_observations[agent_index]
        reward = environment_info.rewards[agent_index]
        done = environment_info.local_done[agent_index]

        return next_state, reward, done

    def on_episode_end(self, environment, agent, network_file):
        agent.save_trained_weights(network_file=network_file)
