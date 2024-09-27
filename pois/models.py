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
            flatten_param_size = 0
            for param in self.mean.parameters():
                flatten_param_size += param.numel()
            if fixed_std:
                self.log_std = torch.ones(flatten_param_size)
            else:
                self.log_std = nn.Parameter(torch.ones(flatten_param_size))
        else: 
            if fixed_std:
                self.log_std = torch.ones(action_dim)
            else:
                self.log_std = nn.Parameter(torch.ones(action_dim))
                
   
    def get_mean(self):
        theta = []
        for param in self.mean.parameters():
            theta.append(param.flatten())
        theta = torch.cat(theta)
        return theta
    
    def get_log_std(self):
        return self.log_std.flatten()
    
    
    def sample_theta(self):
        # get all the parameters from the self.mean layer
        theta = self.get_mean()
        return torch.normal(theta, self.log_std)
    
    def log_prob(self, theta):
        mean = self.get_mean()
        dist = torch.distributions.Normal(mean, self.log_std)
        return dist.log_prob(theta).mean()
    
    def set_theta(self, theta_mean, theta_log_std=None):
        with torch.no_grad():
            if self.fixed_std:
                total_param_size = 0
                for i, param in enumerate(self.mean.parameters()):
                    param_value = theta_mean[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param))
                    total_param_size += param.numel()
            else:
                total_param_size = 0
                for i, param in enumerate(self.mean.parameters()):
                    param_value = theta_mean[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param))
                    total_param_size += param.numel()
                total_param_size = 0   
                for i, param in enumerate(self.log_std):
                    param_value = theta_log_std[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param))
                
    @staticmethod
    def get_layer_gradients(layer):
        grads = []  
        for name, param in layer.named_parameters():
            grads.append(param.grad.flatten())
        return torch.cat(grads)
    
    @staticmethod
    def layer_assign_gradients(layer, grads):
        total_param_size = 0
        for i, param in enumerate(layer.parameters()):
            param_value = grads[total_param_size:total_param_size+param.numel()]
            param.copy_(param_value.view_as(param))
            total_param_size += param.numel()
                
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
        
        
        
class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, fixed_std=True, method='p-pois'):
        super(MLPPolicy, self).__init__()
        self.mean = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25), 
            nn.ReLU(),
            nn.Linear(25, action_dim)
        )
        # initialize mean parameters uniform Xavier
        for layer in self.mean:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.method = method
        self.fixed_std = fixed_std
        if method == 'p-pois':
            flatten_param_size = 0
            for param in self.mean.parameters():
                flatten_param_size += param.numel()
            if fixed_std:
                self.log_std = torch.ones(flatten_param_size)
            else:
                self.log_std = nn.Parameter(torch.ones(flatten_param_size))
        else: 
            if fixed_std:
                self.log_std = torch.ones(action_dim)
            else:
                self.log_std = nn.Parameter(torch.ones(action_dim))
                
    def get_mean(self):
        theta = []
        for param in self.mean.parameters():
            theta.append(param.flatten())
        theta = torch.cat(theta)
        return theta
    
    def get_log_std(self):
        return self.log_std.flatten()
    
    def sample_theta(self):
        # get all the parameters from the self.mean layer
        theta = self.get_mean()
        return torch.normal(theta, self.log_std)
    
    def log_prob(self, theta):
        mean = self.get_mean()
        if torch.isnan(mean).any():
            pdb.set_trace()
        dist = torch.distributions.Normal(mean, self.log_std)
        return dist.log_prob(theta).mean()
    
    def set_theta(self, theta_mean, theta_log_std=None):
        with torch.no_grad():
            if self.fixed_std:
                total_param_size = 0
                for i, param in enumerate(self.mean.parameters()):
                    param_value = theta_mean[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param))
                    total_param_size += param.numel()
            else:
                total_param_size = 0
                for i, param in enumerate(self.mean.parameters()):
                    param_value = theta_mean[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param))
                    total_param_size += param.numel()
                total_param_size = 0   
                for i, param in enumerate(self.log_std):
                    param_value = theta_log_std[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param))
                
    def forward(self, state):
        mean = self.mean(state)
        if self.fixed_std:
            std = torch.ones_like(mean) #self.log_std
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
    
    @staticmethod
    def get_layer_gradients(layer):
        grads = []  
        for name, param in layer.named_parameters():
            grads.append(param.grad.flatten())
        return torch.cat(grads)
    
    @staticmethod
    def layer_assign_gradients(layer, grads):
        total_param_size = 0
        for i, param in enumerate(layer.parameters()):
            param_value = grads[total_param_size:total_param_size+param.numel()]
            param.copy_(param_value.view_as(param))
            total_param_size += param.numel()

model_factory = {
    'linear': GaussianPolicy,
    'mlp': MLPPolicy
}


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
    
