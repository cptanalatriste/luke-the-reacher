from collections import deque

import numpy as np

class TrainingManager():

    def __init__(self, environment_params):
        self.environment_params = environment_params

    def do_reset(self, environment, agent):
        return []

    def keep_going(self):
        True

    def get_action_parameters(self):
        return {}

    def do_step(self, environment, action):
        return ()

    def on_episode_end(self, environment, agent, network_file):
        pass

    def start_training(self, agent, environment, num_episodes, score_window,
                       network_file, target_score):
        all_scores = []
        last_scores = deque(maxlen=score_window)

        for episode in range(1, num_episodes + 1):
            state = self.do_reset(environment, agent)

            current_score = 0.0

            while self.keep_going():
                action_parameters = self.get_action_parameters()
                action = agent.act(state=state,
                                   action_parameters=action_parameters)

                next_state, reward, done = self.do_step(environment, action)
                agent.step(state, action, reward, next_state, done)

                state = next_state
                current_score += reward

                if done:
                    break

            last_scores.append(current_score)
            all_scores.append(current_score)

            average_score = np.mean(last_scores)
            if episode % score_window == 0:
                print("Episode", episode, "Average score over the last", score_window,
                      " episodes: ", average_score)

            if average_score >= target_score:
                print("Environment solved in ", episode + 1, " episodes. ",
                      "Average score: ", average_score)

                agent.save_trained_weights(network_file=network_file)
                break

            self.on_episode_end(environment, agent, network_file)

        return all_scores
