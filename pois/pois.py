import torch
from losses import p_pois_loss, a_pois_loss
from utils import collect_trajectories, line_search
from policy_network import HyperPolicy, GaussianPolicy
import torch.optim as optim
import gym



def train_pois_with_line_search(env, policy_old, policy_new, loss_fn, gamma=1.0, lambda_coef=0.01, num_online_iterations=10, num_offline_iterations=10):
    for j in range(num_online_iterations):
        # Online phase: Collect trajectories using the current policy
        trajectories = collect_trajectories(env, policy_old, num_trajectories=10)

        for k in range(num_offline_iterations):
            # Perform line search to find optimal step size alpha_k
            alpha_k = line_search(policy_new, policy_old, loss_fn, trajectories)
            
            # Perform gradient ascent update with alpha_k
            loss = loss_fn(trajectories, policy_old, policy_new, lambda_coef)
            optimizer = optim.Adam(policy_new.parameters(), lr=alpha_k)  # Use alpha_k as learning rate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the old policy to be the new one
        policy_old.load_state_dict(policy_new.state_dict())
        
        print(f'Online Iteration {j}, Offline Iteration {k}, Loss: {loss.item()}, Alpha: {alpha_k}')


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # A-POIS training
    policy_old = GaussianPolicy(state_dim, action_dim)
    policy_new = GaussianPolicy(state_dim, action_dim)
    train_pois_with_line_search(env, policy_old, policy_new, a_pois_loss)

    # P-POIS training
    hyperpolicy_old = HyperPolicy(state_dim)
    hyperpolicy_new = HyperPolicy(state_dim)
    train_pois_with_line_search(env, hyperpolicy_old, hyperpolicy_new, p_pois_loss)
