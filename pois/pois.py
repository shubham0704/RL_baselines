import torch
from losses import p_pois_loss, a_pois_loss
from utils import collect_trajectories, parabolic_line_search
from policy_network import GaussianPolicy
import torch.optim as optim
import gym
import pdb

import torch

def compute_exact_fisher_information_matrix(hyperpolicy):
    """
    Compute the exact Fisher Information Matrix (FIM) for a Gaussian hyperpolicy.
    The FIM is diagonal, as described in the provided formula.
    """
    mean = hyperpolicy.get_mean()  # Mean of the Gaussian hyperpolicy
    sigma = torch.exp(hyperpolicy.get_log_std())  # Standard deviation (since log_sigma is used)

    # Construct the diagonal Fisher Information Matrix
    F_mu_inv = torch.diag(sigma**2)  # FIM for mean parameters (diag(sigma^2))
    F_sigma_inv = torch.eye(len(sigma)) * 0.5  # FIM for sigma parameters (2 * I)

    return F_mu_inv, F_sigma_inv


def train_p_pois_with_line_search(env, hyperpolicy_old, hyperpolicy_new, 
                                  gamma=1.0, lambda_coef=0.01, 
                                  num_online_iterations=10, 
                                  num_offline_iterations=10,
                                  num_trajectories=10):
    
    for j in range(num_online_iterations):
        # Collect trajectories using old hyperpolicy
        trajectories = []
        thetas = []
        for i in range(num_trajectories):
            theta = hyperpolicy_old.sample_theta()
            hyperpolicy_new.set_theta(theta)  # Set new policy to sample theta
            trajectory = collect_trajectories(env, hyperpolicy_new, 1)
            trajectories += trajectory
            thetas.append(theta)
        
        # Perform offline optimization
        for k in range(num_offline_iterations):
            # Compute Fisher Information Matrix (FIM) for the new hyperpolicy
            F_mean_inv, F_sigma_inv = compute_exact_fisher_information_matrix(hyperpolicy_new)
            
            # Define the P-POIS loss function
            def loss_fn(theta=None):
                if theta is not None:
                    hyperpolicy_new.set_theta(theta)
                return p_pois_loss(trajectories, thetas, 
                                   hyperpolicy_old, hyperpolicy_new, lambda_coef)

            # Compute the gradient of the P-POIS loss
            loss = loss_fn()
            hyperpolicy_new.zero_grad()  # Clear previous gradients
            loss.backward(retain_graph=True)  # Backpropagate the loss

            # Get gradients for mean and log_std
            grad_mean = hyperpolicy_new.mean.weight.grad.flatten()  # Gradient for mean
            grad_sigma = None
            if not hyperpolicy_new.fixed_std:
                grad_sigma = hyperpolicy_new.log_std.grad.flatten()  # Gradient for log_std if learnable

            # Compute natural gradients using FIM
            natural_grad_mean = F_mean_inv @ grad_mean
            natural_grad_sigma = None
            if grad_sigma is not None:
                natural_grad_sigma = F_sigma_inv @ grad_sigma  # Natural gradient for log_std

            # Perform parabolic line search for mean parameters
            alpha_k_mean = parabolic_line_search(
                loss_fn, hyperpolicy_new.get_mean(), grad_mean, F_mean_inv
            )
            
            # If log_std is learnable, perform line search for it as well
            if grad_sigma is not None:
                alpha_k_sigma = parabolic_line_search(
                    loss_fn, hyperpolicy_new.get_log_std(), grad_sigma, F_sigma_inv
                )

            # Update the mean parameters using the natural gradient and alpha_k
            with torch.no_grad():
                mean_update = alpha_k_mean * natural_grad_mean
                weight_update = mean_update.reshape(hyperpolicy_new.mean.weight.shape)
                hyperpolicy_new.mean.weight += weight_update  # Update mean
                if grad_sigma is not None:
                    log_std_update = alpha_k_sigma * natural_grad_sigma
                    log_std_update = log_std_update.reshape(hyperpolicy_new.log_std.shape)
                    hyperpolicy_new.log_std += log_std_update  # Update log_std if learnable

        # Update the old policy to match the new one after offline iterations
        hyperpolicy_old.load_state_dict(hyperpolicy_new.state_dict())
        
        print(f'Online Iteration {j}, Offline Iteration {k}, Loss: {loss.item()}, Alpha_mean: {alpha_k_mean}, Alpha_sigma: {alpha_k_sigma if grad_sigma is not None else None}')


def train_a_pois_with_line_search(env, policy_old, policy_new, 
                                gamma=1.0, lambda_coef=0.01, 
                                num_online_iterations=10, 
                                num_offline_iterations=10,
                                num_trajectories=10):    
    optimizer = torch.optim.Adam(policy_new.parameters(), lr=1e-3)  # Initial learning rate
    
    for j in range(num_online_iterations):
        # Online phase: Collect trajectories using the current policy
        trajectories = collect_trajectories(env, policy_old, num_trajectories=10)

        for k in range(num_offline_iterations):
            # Compute the gradient of the A-POIS loss w.r.t. policy parameters
            def loss_fn(theta):
                return a_pois_loss(trajectories, policy_old, policy_new, lambda_coef)
            optimizer.zero_grad()  # Clear gradients
            loss = loss_fn(policy_new)  # Compute the loss
            loss.backward(retain_graph=True)  # Compute the gradients

            # Get the gradient and initial parameters
            grad_0 = torch.cat([param.grad.view(-1) for param in policy_new.parameters()])  # Concatenate gradients into a single tensor
            theta_0 = torch.cat([param.clone().view(-1) for param in policy_new.parameters()])  # Concatenate parameters into a single tensor
            
            # Define the metric inverse function (identity matrix for simplicity)
            def metric_inv(grad):
                return torch.eye(grad.size(0))  # Identity matrix for simplicity

            # Perform parabolic line search to find optimal step size alpha_k
            alpha_k = parabolic_line_search(loss_fn, theta_0, grad_0, metric_inv)

            # Perform gradient ascent update with the natural gradient and alpha_k
            with torch.no_grad():
                start_idx = 0
                for param in policy_new.parameters():
                    # Update each parameter using the corresponding part of the concatenated grad_0
                    param_shape = param.shape
                    param_size = param.numel()
                    param_grad = grad_0[start_idx:start_idx + param_size].view(param_shape)
                    param.add_(alpha_k * param_grad)
                    start_idx += param_size

        # Update the old policy to match the new one at the end of this iteration
        policy_old.load_state_dict(policy_new.state_dict())
        
        print(f'Online Iteration {j}, Offline Iteration {k}, Loss: {loss.item()}, Alpha: {alpha_k}')


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # A-POIS training
    # policy_old = GaussianPolicy(state_dim, action_dim, method='a-pois')
    # policy_new = GaussianPolicy(state_dim, action_dim, method='a-pois')
    # train_a_pois_with_line_search(env, policy_old, policy_new)

    # P-POIS training
    hyperpolicy_old = GaussianPolicy(state_dim, action_dim, method='p-pois')
    hyperpolicy_new = GaussianPolicy(state_dim, action_dim, method='p-pois')
    train_p_pois_with_line_search(env, hyperpolicy_old, hyperpolicy_new)
