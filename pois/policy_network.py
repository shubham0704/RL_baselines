import torch
import torch.nn as nn
import torch.optim as optim
import gym
import pdb
# Define the Gaussian policy network
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()
        self.mean = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # log of standard deviation (σ)
    
    def forward(self, state):
        mean = self.mean(state)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum()


# Hyperpolicy: Gaussian with mean mu and diagonal covariance sigma^2
class HyperPolicy(nn.Module):
    def __init__(self, theta_dim):
        super(HyperPolicy, self).__init__()
        self.mu = nn.Parameter(torch.zeros(theta_dim))
        self.log_sigma = nn.Parameter(torch.zeros(theta_dim))  # log of standard deviation

    def sample_theta(self):
        std = torch.exp(self.log_sigma)
        return torch.normal(self.mu, std)

    def log_prob(self, theta):
        std = torch.exp(self.log_sigma)
        dist = torch.distributions.Normal(self.mu, std)
        return dist.log_prob(theta).sum()

    def renyi_divergence(self, other_hyperpolicy):
        sigma1 = torch.exp(self.log_sigma)
        sigma2 = torch.exp(other_hyperpolicy.log_sigma)

        term1 = torch.sum((sigma1 / sigma2).pow(2))
        term2 = torch.sum((self.mu - other_hyperpolicy.mu).pow(2) / sigma2.pow(2))
        term3 = 2 * torch.sum(torch.log(sigma2 / sigma1))

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
