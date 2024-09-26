import torch
import torch.nn as nn
import torch.optim as optim
import gym
import pdb
import torch.nn.functional as F
# Define the Gaussian policy network
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, fixed_std=True, method='p-pois'):
        super(GaussianPolicy, self).__init__()
        self.mean = nn.Linear(state_dim, action_dim, bias=False)
        # initialize mean parameters from N(0, 0.01)
        nn.init.normal_(self.mean.weight, 0, 0.01)
        self.method = method
        self.fixed_std = fixed_std
        if method == 'p-pois':
            if fixed_std:
                self.log_std = torch.ones_like(self.mean.weight)
            else:
                self.log_std = nn.Parameter(torch.ones_like(self.mean.weight))
        else: 
            if fixed_std:
                self.log_std = torch.ones(action_dim)
            else:
                self.log_std = nn.Parameter(torch.ones(action_dim))
                
    def get_mean(self):
        return self.mean.weight.flatten()
    
    def get_log_std(self):
        return self.log_std.flatten()
    
    def sample_theta(self):
        return torch.normal(self.mean.weight, self.log_std)
    
    def log_prob(self, theta):
        dist = torch.distributions.Normal(self.mean.weight, self.log_std)
        return dist.log_prob(theta).sum()
    
    def set_theta(self, theta_mean, theta_log_std=None):
        with torch.no_grad():
            if self.fixed_std:
                self.mean.weight.copy_(theta_mean.view_as(self.mean.weight))
            else:
                self.mean.weight.copy_(theta_mean.view_as(self.mean.weight))
                self.log_std.copy_(theta_log_std.view_as(self.mean.weight))
                
    def forward(self, state):
        mean = self.mean(state)
        if self.fixed_std:
            std = self.log_std#torch.ones_like(mean)
        else:
            std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        if self.method == 'p-pois':
            action = mean # p-pois has deterministic action
            return action, 0
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample() # a-pois has stochastic action policy
            return action, dist.log_prob(action).sum()




if __name__ == "__main__":
    # define the cartpole environment
    env = gym.make("CartPole-v1")
    # Initialize the policy
    state_dim = env.observation_space.shape[0]
    # check if the action space is discrete or continuous
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n # Discrete actions
    else:
        action_dim = env.action_space.shape[0] # Continuous actions
    policy = GaussianPolicy(state_dim, action_dim)
