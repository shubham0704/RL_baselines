import torch
import torch.nn as nn
import torch.optim as optim
import gym
import pdb
import torch.nn.functional as F
import numpy as np

# create a base class for the policy network which takes in the mean function
class PolicyNetwork(nn.Module):
    def __init__(self, mean: nn.Module, method: str, 
                 fixed_std: bool, action_dim: int, device: torch.device):
        super(PolicyNetwork, self).__init__()
        self.mean = mean
        self.method = method
        self.fixed_std = fixed_std
        self.action_dim = action_dim
        if method == 'p-pois':
            flatten_param_size = 0
            for param in self.mean.parameters():
                flatten_param_size += param.numel()
            if fixed_std:
                self.log_std = torch.ones(flatten_param_size).to(device)
            else:
                self.log_std = nn.Parameter(torch.ones(flatten_param_size).to(device))
        else: 
            if fixed_std:
                self.log_std = torch.ones(action_dim).to(device)
            else:
                self.log_std = nn.Parameter(torch.ones(action_dim).to(device))
                
   
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
                    param.copy_(param_value.view_as(param).clone())
                    total_param_size += param.numel()
            else:
                total_param_size = 0
                for i, param in enumerate(self.mean.parameters()):
                    param_value = theta_mean[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param).clone())
                    total_param_size += param.numel()
                total_param_size = 0   
                for i, param in enumerate(self.log_std):
                    param_value = theta_log_std[total_param_size:total_param_size+param.numel()]
                    param.copy_(param_value.view_as(param).clone())
                
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
            param.add_(param_value.view_as(param).detach().clone())
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
        
    def predict(self, observations, states=None, episode_starts=None, deterministic=True):
        """
        Predict actions for given observations.
        :param observations: (np.ndarray or torch.Tensor) The input observations.
        :param states: (Optional) The last states (used in recurrent policies).
        :param episode_starts: (Optional) Indicates the start of episodes.
        :param deterministic: (bool) Whether to use deterministic actions.
        :return: (np.ndarray, Optional) The predicted actions and the next states.
        """
        # Convert observations to torch tensors if they are not already
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(observations)
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)  # Add batch dimension if needed

        # Pass observations through the network to get mean and std
        with torch.no_grad():
            mean, std = self.forward(observations)

        # Determine action based on deterministic flag and method
        if deterministic or self.method == 'p-pois':
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        action = torch.argmax(action, dim=1)
        # Convert action to numpy array
        action = action.detach().cpu().numpy()
        # Ensure action is a 1D array, especially when using vectorized environments
        action = np.atleast_1d(action)
        
        return action, states
    

# Define the Gaussian policy network
class GaussianPolicy(PolicyNetwork):
    def __init__(self, state_dim, action_dim, fixed_std=True, method='p-pois'):
        mean = nn.Linear(state_dim, action_dim, bias=False)
        # initialize mean parameters from N(0, 0.01)
        nn.init.normal_(mean.weight, 0, 0.01)
        device = torch.device('cpu')
        super(GaussianPolicy, self).__init__(mean, method, fixed_std, action_dim, device)
        
        
class MLPPolicy(PolicyNetwork):
    def __init__(self, state_dim, action_dim, fixed_std=True, method='p-pois'):
        mean = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25), 
            nn.ReLU(),
            nn.Linear(25, action_dim)
        )
        # initialize mean parameters uniform Xavier
        for layer in mean:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        device = torch.device('cpu')
        super(MLPPolicy, self).__init__(mean, method, fixed_std, action_dim, device)
        
        
class VAEPolicy(PolicyNetwork):
    def __init__(self, state_dim, action_dim, fixed_std=True, method='p-pois'):
        """
        State: x
        Action: [mean(x), std(x) (q(z|x) ~ N(mean(x), std(x))), \hat x (reconstruction output of VAE)]
        """
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        
        self.mean = nn.Linear(50, action_dim)
        self.log_std = nn.Linear(50, action_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(action_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim)
        )
        
        device = torch.device('cpu')
        super(VAEPolicy, self).__init__(self.mean, method, fixed_std, action_dim, device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, state):
        encoded = self.encoder(state)
        mean = self.mean(encoded)
        log_std = self.log_std(encoded)
        std = torch.exp(log_std)
        return mean, std
    
    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()  # Use reparameterization trick
        reconstruction = self.decoder(z)
        
        if self.method == 'p-pois':
            action = torch.cat([mean, std, reconstruction], dim=-1)
            return action, 0
        else:
            action = torch.cat([z, reconstruction], dim=-1)
            return action, dist.log_prob(z).sum()
    
    def predict(self, observations, states=None, episode_starts=None, deterministic=True):
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(observations)
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.forward(observations)
            if deterministic or self.method == 'p-pois':
                z = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                z = dist.sample()
            reconstruction = self.decoder(z)
            action = torch.cat([z, reconstruction], dim=-1)
        
        action = action.detach().cpu().numpy()
        return np.atleast_1d(action), states
    
    def get_mean(self):
        return torch.cat([param.flatten() for param in self.parameters()])
    
    def set_theta(self, theta_mean, theta_log_std=None):
        with torch.no_grad():
            start = 0
            for param in self.parameters():
                end = start + param.numel()
                param.copy_(theta_mean[start:end].view_as(param))
                start = end

# Update the model_factory dictionary
model_factory = {
    'linear': GaussianPolicy,
    'mlp': MLPPolicy,
    'vae': VAEPolicy
}