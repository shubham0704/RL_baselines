import torch

def p_pois_loss(trajectories, thetas, hyperpolicy_old, hyperpolicy_new, lambda_coef):
    N = len(trajectories)
    total_loss = 0
    wt_sq_sum = 0
    for traj, theta in zip(trajectories, thetas):
        G = 0
        # Calculate return and importance weight
        for state, action, reward, _ in reversed(traj):
            # the above has log_action_prob
            # this trajectory should also have theta that
            # was sampled from the hyperpolicy
            G = reward + G  # No discounting (gamma = 1 for simplicity)
            # Calculate the importance weight
            log_prob_new = hyperpolicy_new.log_prob(theta)
            log_prob_old = hyperpolicy_old.log_prob(theta) # log_prob_theta
            w_rho = torch.exp(log_prob_new - log_prob_old)
            wt_sq_sum += w_rho.pow(2)
            weighted_return = w_rho * G
            total_loss += weighted_return
   
    # Calculate Renyi divergence between hyperpolicies
    ess = N / wt_sq_sum
    ess_penalty = lambda_coef / torch.sqrt(ess)

    # Surrogate loss with Renyi divergence penalty
    loss = total_loss / N - lambda_coef * ess_penalty
    
    return loss



def a_pois_loss(trajectories, policy_old, policy_new, lambda_coef):
    N = len(trajectories)
    total_loss = 0
    wt_sq_sum = 0
    # G is the return over the trajectory
    for traj in trajectories:
        G = 0
        # Compute the weighted return and Renyi divergence for each trajectory
        for state, action, reward, log_prob_old in reversed(traj):
            G = reward + G  # No discounting (gamma = 1 for simplicity)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean_new, std_new = policy_new(state_tensor)
            dist_new = torch.distributions.Normal(mean_new, std_new)
            log_prob_new = dist_new.log_prob(action).sum()
            w_theta = torch.exp(log_prob_new - log_prob_old)  # Importance weight
            wt_sq_sum += w_theta.pow(2)
            weighted_return = w_theta * G
            total_loss += weighted_return
    
    # Calculate Renyi divergence between hyperpolicies
    ess = N / wt_sq_sum
    ess_penalty = lambda_coef / torch.sqrt(ess)
    # Surrogate loss with Renyi divergence penalty
    loss = total_loss / N - lambda_coef * ess_penalty
    return loss

