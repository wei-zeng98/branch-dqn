import torch.nn as nn
import torch
import math
import random
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class BranchDuelingDQN(nn.Module):
    """
    input: (batch_size, obs_dim)
    output: (batch_size, act_dim, act_sub_dim)
    """

    def __init__(self, observation_space, action_space, act_sub_dim, 
                 device, act_type, hidden_sizes=(64, 64), activation=nn.ReLU):
        super().__init__()
        self.device = device
        obs_dim     = observation_space.shape[0]
        
        if act_type == 'discrete':
            act_dim = 1
        elif act_type == 'box':
            act_dim = action_space.shape[0]
        
        self.feature   = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.value     = nn.Linear(hidden_sizes[-1], 1)
        self.adv_layer = []

        for i in range(act_dim):
            adv = nn.Linear(hidden_sizes[-1], act_sub_dim[i]).to(device)
            self.adv_layer.append(adv)
        
        self.steps_done = 0
        self.idx_l = np.zeros(act_dim)
        self.idx_h = np.array(act_sub_dim)
    
    def forward(self, x):
        # calculate value and advantage for each action
        feature = self.feature(x)
        value   = self.value(feature)
        adv_val = []
        q       = torch.FloatTensor().to(self.device)
        
        for adv_layer_i in self.adv_layer:
            adv_val.append(adv_layer_i(feature))
        
        # calculate action value   
        for adv_i in adv_val:
            q = torch.cat((q, (value + adv_i - adv_i.mean()).unsqueeze(0)), 0)

        # return size: (batch_size, act_dim, act_sub_dim)
        return q.permute(1, 0, 2)
    
    def act(self, obs, deterministic=False, es=0.9, ee=0.05, ed=1000):
        """
        Return action index
        es: epsilon start value, ee: epsilon end value, ed: epsilon decay value
        """

        sample = random.random()
        eps_threshold = ee + (es - ee) * \
            math.exp(-1. * self.steps_done / ed)
        self.steps_done += 1
        
        if sample > eps_threshold or deterministic:
            # pick the action with largest value
            with torch.no_grad():
                q = self.forward(obs.unsqueeze(0)).squeeze(0)
                return torch.argmax(q, 1).cpu().numpy()
        
        else:
            # pick a random action
            return np.random.randint(self.idx_l, self.idx_h)