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
        self.mean = nn.Linear(state_dim, action_dim)
        # initialize mean parameters from N(0, 0.01)
        nn.init.normal_(self.mean.weight, 0, 0.01)
        self.method = method
        self.fixed_std = fixed_std
        if fixed_std:
            self.log_std = torch.ones(action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def sample_theta(self):
        std = torch.exp(self.log_std)
        return torch.normal(self.mu, std)
    
    def forward(self, state):
        mean = self.mean(state)
        if self.fixed_std:
            std = self.log_std#torch.ones_like(mean)
        else:
            std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        if self.method == 'p-pois':
            action = mean # p-pois has deterministic action
        else:
            action = dist.sample() # a-pois has stochastic action policy
        return action, dist.log_prob(action).sum()

    def renyi_divergence(self, other_hyperpolicy):
        std1 = torch.exp(self.log_std)
        std2 = torch.exp(other_hyperpolicy.log_std)
        term1 = torch.sum((std1 / std2).pow(2))
        term2 = torch.sum((self.mu - other_hyperpolicy.mu).pow(2) / std2.pow(2))
        term3 = 2 * torch.sum(torch.log(std2 / std1))

        return 0.5 * (term1 + term2 - term3 - len(self.mu))




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
