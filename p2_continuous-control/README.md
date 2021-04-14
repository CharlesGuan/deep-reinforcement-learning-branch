[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, the environment [Reacher] is used.  
(https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, the version contains 20 identical agents, each with its own copy of the environment.  The version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience. 

The barrier for solving the version of the environment is to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved if a reward of +100 is obtain for 30 consecutive episodes.

The method to use is an actor-critic algorithm, the Deep Deterministic Policy Gradients (DDPG) algorithm.

## Getting started
1. To set up your Python environment correctly follow [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies).

2. To download the Unity Environment follow:
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) for Linux
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) for OSX
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip) for Microsoft 32 bits
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip) for Microsoft 64 bits
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

3. Then, place the file in the p2_continuous-control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

4. The training can be run directly in the notebook

### Files
There are 3 major files in this project: model, agent and continuous_control as explained below:

1. model.py contains the architecture for actor and critic networks 
2. agent.py File containing the actor/critic classes, which produce continuous actions given a specific state, update four networks (actor and critic vs local and target), and buffer learning experiences (sate, action reward, next state) for efficient training. 
3. Continuous_Control.ipynb instantiates the agent and manages the interactions between the Agent class and the environment.

