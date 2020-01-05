# luke-the-reacher
A deep reinforcement-learning agent for a double-jointed robotic arm.

## Project Details:
Luke-the-reacher is a deep-reinforcement learning agent designed for the Reacher environment from the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

![The Reacher environment](https://github.com/cptanalatriste/luke-the-reacher/blob/master/img/environment.png?raw=true)

The state is represented via a vector of 33 elements. They correspond to the position, rotation, velocity, and angular velocities of the double-jointed arm. The agent's actions are composed by vectors of 4 real-valued elements between -1 and 1.
These values represent the torque to apply to its two joints.

The agent is rewarded with +0.1 points every time step the arm is in contact with the target. We consider our agent has mastered the task when he reaches an **average score of 30, over 100 episodes**.

## Getting Started
Before running your agent, be sure to accomplish this first:
1. Clone this repository.
1. Download the reacher environment appropriate to your operating system (available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)). Be sure to select the file corresponding to **Version 1: One(1) Agent**.
1. Place the environment file in the cloned repository folder.
1. Setup an appropriate Python environment. Instructions available [here.]
(https://github.com/udacity/deep-reinforcement-learning)

##  Instructions
You can start running and training the agent by exploring `Navigation.ipynb`. Also available in the repository:

* `luke_reacher.py` contains the agent code.
* `reacher_manager.py` has the code for training the agent.

