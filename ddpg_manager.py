class DdpgManager():

    def __init__(self, environment_params):
        self.environment_params = environment_params

    def start_training(self, agent, environment, num_episodes, score_window,
                       network_file, target_score):
        all_scores = []
        return all_scores
