import torch

def p_pois_loss(trajectories, hyperpolicy_old, hyperpolicy_new, lambda_coef):
    N = len(trajectories)
    total_loss = 0
    total_divergence = 0
    
    for traj in trajectories:
        theta = hyperpolicy_new.sample_theta()  # Sample policy parameters from new hyperpolicy
        G = 0
        weighted_return = 0
        
        # Calculate return and importance weight
        for state, action, reward, log_prob_old in reversed(traj):
            G = reward + G  # No discounting (gamma = 1 for simplicity)

        # Calculate the importance weight
        log_prob_old = hyperpolicy_old.log_prob(theta)
        log_prob_new = hyperpolicy_new.log_prob(theta)
        w_rho = torch.exp(log_prob_new - log_prob_old)
        
        weighted_return += w_rho * G
        total_loss += weighted_return

    # Calculate Renyi divergence between hyperpolicies
    total_divergence = hyperpolicy_new.renyi_divergence(hyperpolicy_old)

    # Surrogate loss with Renyi divergence penalty
    loss = total_loss / N - lambda_coef * torch.sqrt(total_divergence / N)
    
    return loss



def a_pois_loss(trajectories, policy_old, policy_new, lambda_coef):
    N = len(trajectories)
    total_loss = 0
    total_divergence = 0
    
    for traj in trajectories:
        weighted_return = 0
        renyi_sum = 0
        G = 0
        
        # Compute the weighted return and Renyi divergence for each trajectory
        for state, action, reward, log_prob_old in reversed(traj):
            G = reward + G  # No discounting (gamma = 1 for simplicity)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean_new, std_new = policy_new(state_tensor)
            dist_new = torch.distributions.Normal(mean_new, std_new)
            log_prob_new = dist_new.log_prob(action).sum()
            
            w_theta = torch.exp(log_prob_new - log_prob_old)  # Importance weight
            weighted_return += w_theta * G
            
            # Renyi divergence approximation between new and old policy
            mean_old, std_old = policy_old(state_tensor)
            renyi_sum += torch.sum((std_new / std_old).pow(2) + (mean_new - mean_old).pow(2) / std_old.pow(2))

        total_loss += weighted_return
        total_divergence += renyi_sum
    
    # Surrogate loss with Renyi divergence penalty
    loss = total_loss / N - lambda_coef * torch.sqrt(total_divergence / N)
    
    return loss

