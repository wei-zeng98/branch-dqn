This is a PyTorch implementation of Dueling DQN with action branching architectures for [Gym](https://gym.openai.com/) environment with `Discrete` and `Box` action space.

# Action Branching Architectures

Reference paper: https://arxiv.org/abs/1711.08946

The branching architecture is summarized as following.

![Action Branching Architectures.](./img/Action-Branching-Architectures.png)

For action of `Box` class, certain number of sub-actions need to be sampled from each continuous action dimension by setting parameter `sub_act_dim`, which could either be an integer or a list/tuple in the size of action dimension indication number of sub-actions for each action dimension. Given `sub_act_dim`, actions including lowest and highest possible value are equally sample from each action dimension as following.

![Sub-action sampling.](./img/sub-actions.png)

# Train

## Requirements

python=3.8.11

torch=1.9.0

gym=0.19.0

tensorboard=2.8.0

## Start Training

Simply run `python dqn.py`.

## Results

CartPole-v1 (discrete action) and MountainCarContinuous-v0 (continuous action) of Gym environment are tested, episode return are show in the following respectively. After around 1.6k and 120 episodes for each case, the agent start to gain steady rewards.

![Episode return of CartPole-v1 in training process.](./img/CartPole-v1.png)

![Episode return of MountainCarContinuous-v0 in training process.](./img/MountainCarContinuous-v0.png)