import torch
import pdb
def collect_trajectories(env, policy, num_trajectories):
    trajectories = []
    for _ in range(num_trajectories):
        state, _ = env.reset()
        traj = []
        done = False
        while not done or truncated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = policy.sample_action(state_tensor)
            next_state, reward, done, truncated, _ = env.step(torch.argmax(action.detach()).item())
            traj.append((state, action, reward, log_prob))
            state = next_state
        trajectories.append(traj)
    return trajectories

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
    grad = torch.autograd.grad(current_loss, policy.parameters(), create_graph=True)
    
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

