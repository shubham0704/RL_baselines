import torch
import pdb
def p_pois_loss(trajectories, thetas, hyperpolicy_old, hyperpolicy_new, lambda_coef):
    N = len(trajectories)
    total_loss = 0
    wt_sum = 0
    all_weights = []

    # First, accumulate all raw importance weights for normalization
    for traj, theta in zip(trajectories, thetas):
        G = 0
        for state, action, reward, _ in reversed(traj):
            G = reward + G  # No discounting (gamma = 1 for simplicity)
            # if any nan in theta set trace
            if torch.isnan(theta).any():
                pdb.set_trace()
            # Calculate the importance weight
            log_prob_new = hyperpolicy_new.log_prob(theta)
            log_prob_old = hyperpolicy_old.log_prob(theta)
            w_rho = torch.clamp(torch.exp(log_prob_new - log_prob_old), min=1e-6, max=1e2)
            all_weights.append(w_rho)
            wt_sum += w_rho

    # Normalize the importance weights
    normalized_weights = [w_rho / (wt_sum + 1e-8) for w_rho in all_weights]
    # Now, compute the loss using normalized weights
    idx = 0
    for traj, theta in zip(trajectories, thetas):
        G = 0
        for state, action, reward, _ in reversed(traj):
            G = reward + G

            # Use the normalized importance weight
            w_rho_normalized = normalized_weights[idx]
            idx += 1

            # Weighted return using normalized importance weights
            weighted_return = w_rho_normalized * G
            total_loss += weighted_return

    # Calculate Renyi divergence between hyperpolicies (Effective Sample Size, ESS)
    ess = N / (wt_sum + 1e-8)
    ess_penalty = lambda_coef / (torch.sqrt(ess) + 1e-5)

    # Cap the penalty to avoid too high values
    ess_penalty = torch.where(ess_penalty > 1e5, torch.tensor(1e5, device=ess_penalty.device), ess_penalty)
    print("ess_penalty", ess_penalty)
    print("total_loss", total_loss)
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
    ess_penalty = lambda_coef / (torch.sqrt(ess) + 1e-5)
    ess_penalty = torch.where(ess_penalty > 1e5, torch.tensor(1e5), ess_penalty)
    # Surrogate loss with Renyi divergence penalty
    loss = total_loss / N - lambda_coef * ess_penalty
    return loss

