from core import BranchDuelingDQN
import torch
from torch.autograd import Variable
import numpy as np
from collections import namedtuple, deque
import random
import time
from tensorboardX import SummaryWriter
from gym.spaces import Discrete, Box

Transition = namedtuple('Transition',
                	    ('obs', 'action', 'next_obs', 'reward', 'done'))

class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return Transition(*zip(*random.sample(self.memory, batch_size)))
    
    def get_len(self):
        return len(self.memory)

def dqn(env_fn, max_steps_per_epoch=35040, epochs=20, gamma=0.999, sub_act_dim=21, 
        criterion=torch.nn.MSELoss(), batch_size=64, freq_target_update=500, lr=1e-4):
    
    env = env_fn

    if isinstance(env.action_space, Discrete):
        act_type    = 'discrete'
        sub_act_dim = [env.action_space.n]
        act_dim     = 1
        low         = [0]
        seg         = [1]
    
    elif isinstance(env.action_space, Box):
        act_type    = 'box'
        act_dim  = env.action_space.shape[0]
        if isinstance(sub_act_dim, int):
            sub_act_dim = [sub_act_dim] * act_dim
        else:
            assert len(sub_act_dim) == act_dim, "Action subdimension does not match!"
        high        = env.action_space.high
        low         = env.action_space.low
        seg         = (high - low) / (np.array(sub_act_dim) - 1)
    
    else:
        raise NotImplementedError('This type of action is not implemented.')
    
    buffer     = ReplayBuffer(int(1e6))
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = BranchDuelingDQN(env.observation_space, env.action_space, 
                                  sub_act_dim, device, act_type).to(device)
    target_net = BranchDuelingDQN(env.observation_space, env.action_space, 
                                  sub_act_dim, device, act_type).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr)
    writer = SummaryWriter()

    def get_action_idx(o, deterministic=False):
        o = torch.as_tensor(o, dtype=torch.float32).to(device)
        act_idx = policy_net.act(o, deterministic)
        return act_idx
    
    def update_target():
        """To update the parameters of target network with policy network"""
        target_net.load_state_dict(policy_net.state_dict())
    
    def compute_td_loss():
        """Compute TD error, update in experience buffer and learn a step in policy_net"""
        
        data = buffer.sample(batch_size)
        obs      = Variable(torch.FloatTensor(np.array(data.obs))).to(device)       # (batch_size, obs_dim)
        next_obs = Variable(torch.FloatTensor(np.array(data.next_obs))).to(device)  # (batch_size, obs_dim)
        action   = Variable(torch.LongTensor(np.array(data.action))).to(device)     # (batch_size, act_dim)
        reward   = Variable(torch.FloatTensor(np.array(data.reward))).to(device)    # (batch_size)
        done     = Variable(torch.IntTensor(np.array(data.done))).to(device)        # (batch_size)
        
        q_policy_all = policy_net(obs)                                              # (batch_size, act_dim, sub_act_dim)
        
        with torch.no_grad():
            next_q_policy_all = policy_net(next_obs)
            next_q_target_all = target_net(next_obs)

        # get predicted values from policy_net and action
        q_policy = q_policy_all.gather(2, action.unsqueeze(2)).squeeze(2)           # (batch_size, act_dim)                                           

        # get the max value action index from policy_net
        a_idx = torch.max(next_q_policy_all, 2)[1].unsqueeze(2)                     # (batch_size, act_dim, 1)

        # get target values from target_net and max value action
        next_q_target = (next_q_target_all.gather(2, a_idx).squeeze(2)).mean(1)     # (batch_size)
        expected_q_target = reward + gamma * next_q_target * (1 - done)             # (batch_size)
        expected_q_target = expected_q_target.unsqueeze(1)                          # (batch_size, 1)                                 
        expected_q_target = expected_q_target.repeat(1, act_dim)                    # (batch_size, act_dim)                 
        
        # return loss
        return criterion(q_policy, expected_q_target)
    
    def update():
        policy_net.zero_grad()
        loss = compute_td_loss()
        loss.backward()
        optimizer.step()
        return loss.detach().item()
    
    # Prepare for interaction with environment
    total_steps = max_steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    epoch = 0
    steps_per_epoch = 0
    print('  -------- Begin Training --------  ')
    print('Epoch\tEpRet\tEpLen\tTimeSteps')

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        a_idx = get_action_idx(o)
        a = low + a_idx * seg
        if act_type == 'discrete':
            # for Discrete action, action is an integer
            a = a[0]
        
        o2, r, d, _ = env.step(a)
        buffer.push(o, a_idx, o2, r, d)
        writer.add_scalar('Train/Reward', r, t)
        
        o = o2
        ep_ret += r
        ep_len += 1
        steps_per_epoch += 1
        
        if t % freq_target_update == 0:
            update_target()
        
        if buffer.get_len() > batch_size:
            loss = update()
            writer.add_scalar('Train/Loss', loss, t)
        
        if d or steps_per_epoch > max_steps_per_epoch:
            print('{}\t{}\t{}\t{}'.format(epoch, ep_ret, ep_len, t))
            writer.add_scalar('Train/Return', ep_ret, epoch)
            o, ep_ret, ep_len = env.reset(), 0, 0
            epoch += 1
            steps_per_epoch = 0

if __name__ == '__main__':
    import gym
    env = gym.make('MountainCarContinuous-v0')
    dqn(env)