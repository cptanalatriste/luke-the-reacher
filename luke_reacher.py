import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dqn_utils import ReplayBuffer, OUNoise, update_model_parameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CriticNetwork(nn.Module):

    def __init__(self, state_size, action_size, first_layer_output,
                 second_layer_output):
        super(CriticNetwork, self).__init__()

        first_layer_input = state_size + action_size

        self.first_linear_layer = nn.Linear(in_features=first_layer_input,
                                            out_features=first_layer_output)

        self.second_linear_layer = nn.Linear(in_features=first_layer_output,
                                             out_features=second_layer_output)
        self.third_linear_layer = nn.Linear(in_features=second_layer_output,
                                            out_features=1)

    def forward(self, state, action):
        data_in_transit = torch.cat((state, action), dim=1)
        data_in_transit = F.relu(self.first_linear_layer(data_in_transit))
        data_in_transit = F.relu(self.second_linear_layer(data_in_transit))

        return self.third_linear_layer(data_in_transit)


class ReacherAgent():

    def __init__(self, state_size, action_size, action_min=-1, action_max=1,
                 buffer_size=int(1e5), min_learning_samples=128,
                 actor_learning_rate=1e-4, actor_2nd_input=400,
                 actor_2nd_output=300, critic_1st_output=400, critic_2nd_output=300,
                 critic_learning_rate=1e-3, gamma=0.9, tau=1e-3):

        self.state_size = state_size
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max
        self.min_learning_samples = min_learning_samples
        self.gamma = gamma
        self.tau = tau

        self.actor_local_network = self.get_actor_network(second_layer_input=actor_2nd_input,
                                                          second_layer_output=actor_2nd_output)
        self.actor_target_network = self.get_actor_network(second_layer_input=actor_2nd_input,
                                                           second_layer_output=actor_2nd_output)
        update_model_parameters(tau=1.0, local_network=self.actor_local_network,
                                target_network=self.actor_target_network)
        self.actor_optimizer = optim.Adam(self.actor_local_network.parameters(),
                                          lr=actor_learning_rate)

        self.critic_local_network = self.get_critic_network(first_layer_output=critic_1st_output,
                                                            second_layer_output=critic_2nd_output)
        self.critic_target_network = self.get_critic_network(first_layer_output=critic_1st_output,
                                                             second_layer_output=critic_2nd_output)
        update_model_parameters(tau=1.0, local_network=self.critic_local_network,
                                target_network=self.critic_target_network)
        self.critic_optimizer = optim.Adam(self.critic_local_network.parameters(),
                                           lr=critic_learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          action_type=np.float32,
                                          training_batch_size=min_learning_samples,
                                          device=DEVICE)
        self.noise_generator = OUNoise(size=action_size)

    def get_actor_network(self, second_layer_input, second_layer_output):
        model = nn.Sequential(
            nn.Linear(in_features=self.state_size, out_features=second_layer_input),
            nn.ReLU(),
            nn.Linear(in_features=second_layer_input, out_features=second_layer_output),
            nn.ReLU(),
            nn.Linear(in_features=second_layer_output, out_features=self.action_size),
            nn.Tanh())

        return model.to(DEVICE)

    def get_critic_network(self, first_layer_output, second_layer_output):
        model = CriticNetwork(state_size=self.state_size,
                              action_size=self.action_size,
                              first_layer_output=first_layer_output,
                              second_layer_output=second_layer_output)

        return model.to(DEVICE)

    def act(self, state, action_parameters):

        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.actor_local_network.eval()
        with torch.no_grad():
            action = self.actor_local_network(state).numpy()
        self.actor_local_network.train()

        add_noise = action_parameters['add_noise']
        if add_noise:
            action += torch.from_numpy(self.noise_generator.sample()).float()

        return np.clip(action, self.action_min, self.action_max)

    def step(self, state, action, reward, next_state, done):
        """
        Receives information from the environment. Is in charge of storing
        experiences in the replay buffer and triggering agent learning.
        """

        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) > self.min_learning_samples:

            learning_samples = self.replay_buffer.sample()
            self.learn(learning_samples)

    def learn(self, learning_samples):
        states, actions, rewards, next_states, dones = learning_samples

        next_actions = self.actor_target_network(next_states)
        q_values_next_state = self.critic_target_network(next_states,
                                                         next_actions.detach())
        q_value_current_state = rewards + (self.gamma * q_values_next_state * (1 - dones))
        q_value_expected = self.critic_local_network(states, actions)

        critic_loss = F.mse_loss(input=q_value_expected,
                                 target=q_value_current_state)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = self.actor_local_network(states)
        actor_loss = -self.critic_local_network(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        update_model_parameters(tau=self.tau,
                                local_network=self.critic_local_network,
                                target_network=self.critic_target_network)
        update_model_parameters(tau=self.tau,
                                local_network=self.actor_local_network,
                                target_network=self.actor_target_network)

    def reset(self):
        self.noise_generator.reset()

    def save_trained_weights(self, network_file):
        actor_network_file = "actor_" + network_file
        torch.save(self.actor_local_network.state_dict(), actor_network_file)

        critic_network_file = "critic_" + network_file
        torch.save(self.critic_local_network.state_dict(), critic_network_file)
