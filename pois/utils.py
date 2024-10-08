import torch
import pdb
import gym
def collect_trajectories(env, policy, episodes_per_iteration, device='cpu'):
    trajectories = []
    returns = []
    device = torch.device(device)
    for _ in range(episodes_per_iteration):
        state, _ = env.reset()
        traj = []
        done = False
        truncated = False
        episode_return = 0.0  # Initialize return for the episode
        while not (done or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, log_prob = policy.sample_action(state_tensor)
            action_value = torch.argmax(action.detach()).item() if isinstance(env.action_space, gym.spaces.Discrete) else action.detach().numpy()
            next_state, reward, done, truncated, _ = env.step(action_value)
            traj.append((state, action, reward, log_prob))  # Log the transition
            state = next_state
            episode_return += reward  # Accumulate reward
        trajectories.append(traj)
        returns.append(episode_return)  # Store return for the episode
    return trajectories, returns

def line_search(policy, policy_old, loss_fn, trajectories, initial_alpha=1.0, c1=1e-4, tau=0.5, max_iters=10):
    """
    Perform a simple line search to find the optimal step size alpha_k.
    
    Args:
    - policy: The current policy (parameterized by theta_k).
    - policy_old: The old policy used to collect trajectories.
    - trajectories: Collected trajectories.
    - initial_alpha: Initial guess for the step size.
    - c1: Armijo condition constant.
    - tau: Backtracking factor (reduce step size by this factor each time).
    - max_iters: Maximum number of line search iterations.
    
    Returns:
    - alpha_k: The optimal step size found using line search.
    """
    alpha_k = initial_alpha
    current_loss = loss_fn(trajectories, policy_old, policy, lambda_coef=0.01)
    grad = torch.autograd.grad(current_loss, policy.parameters(), 
                               create_graph=True, retain_graph=True)
    
    # Compute the direction d_k (negative gradient direction)
    direction = [-g for g in grad]
    
    # Initial loss value at current parameters
    for _ in range(max_iters):
        # Temporarily update the parameters
        with torch.no_grad():
            for param, dir_step in zip(policy.parameters(), direction):
                param.add_(alpha_k * dir_step)
        
        # Compute new loss with the updated parameters
        new_loss = loss_fn(trajectories, policy_old, policy, lambda_coef=0.01)
        
        # Armijo sufficient decrease condition
        if new_loss <= current_loss + c1 * alpha_k * sum((g * d).sum() for g, d in zip(grad, direction)):
            return alpha_k  # Accept the current step size
        
        # If not sufficient, reduce alpha_k
        alpha_k *= tau
    
    return alpha_k  # Return final alpha_k after max_iters



def parabolic_line_search(loss_fn, theta_0, grad_0, metric_inv, 
                          max_ls_steps=30, tol_delta_L=1e-4, eta=0.5,
                          min_alpha=1e-10, max_alpha=1.0):
    """
    Parabolic Line Search for finding optimal step size.

    Args:
    - loss_fn: Callable that returns loss for a given theta.
    - theta_0: Initial policy parameters (tensor).
    - grad_0: Initial gradient of the loss w.r.t. parameters (tensor).
    - metric_inv: Inverse of the metric (can be identity if unavailable).
    - max_ls_steps: Maximum number of line search iterations.
    - tol_delta_L: Tolerance for loss improvement.
    - eta: Constant for adjusting step size.
    - min_alpha: Minimum allowed step size.
    - max_alpha: Maximum allowed step size.

    Returns:
    - alpha_opt: Optimal step size found by the line search.
    """
    alpha_0 = 0
    epsilon_1 = 1
    delta_L_k_1 = -float('inf')
    L_0 = loss_fn(theta_0).item()
    if not isinstance(metric_inv, torch.Tensor):
        metric_inv = torch.eye(theta_0.shape[0])
    
    for l in range(max_ls_steps):
        # Step size calculation with overflow protection
        norm_grad = torch.clamp((grad_0 * (metric_inv @ grad_0)).sum(dim=-1), min=1e-8)
        alpha_l = torch.clamp(epsilon_1 / norm_grad, min=min_alpha, max=max_alpha)
        
        # Update parameters based on the current step size
        theta_l = theta_0 - alpha_l * (metric_inv @ grad_0)
        # Compute new loss and the loss difference
        L_l = loss_fn(theta_l).item()
        delta_L_l = L_l - L_0
        # Check if the improvement is within tolerance
        if delta_L_l < delta_L_k_1 + tol_delta_L:
            return alpha_0 if l == 0 else alpha_l

        # Update epsilon based on improvement
        if delta_L_l > (epsilon_1 * (1 - 2 * eta)) / (2 * eta):
            epsilon_1 = eta * epsilon_1
        else:
            denominator = max(2 * (epsilon_1 - delta_L_l), 1e-8)
            epsilon_1 = epsilon_1**2 / denominator

        delta_L_k_1 = delta_L_l
        alpha_0 = alpha_l

    return alpha_0  # Return the last alpha value if max iterations are reached
