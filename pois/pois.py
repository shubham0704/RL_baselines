from learn import train_p_pois_with_line_search, train_a_pois_with_line_search
from models import model_factory
import gym
import torch



class APOIS:
    def __init__(self, env, model_type='mlp', save_path='./policy.pth'):
        self.env = env
        self.model_type = model_type
        self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box): # continuous action space
            self.action_dim = env.action_space.shape[0] 
        else: # discrete action space
            self.action_dim = env.action_space.n
        self.policy_old = model_factory[model_type](self.state_dim, self.action_dim)
        self.policy_new = model_factory[model_type](self.state_dim, self.action_dim)
        self.save_path = save_path
    def learn(self, lambda_coef,num_trajectories=10, 
              num_online_iterations=10, num_offline_iterations=10):
        self.policy_new = train_a_pois_with_line_search(self.env, 
                                                        self.policy_old, 
                                                        self.policy_new, 
                                                        lambda_coef=lambda_coef,
                                                        num_offline_iterations=num_offline_iterations,
                                                        num_online_iterations=num_online_iterations,
                                                        num_trajectories=num_trajectories)
        # save the new policy
        torch.save(self.policy_new.state_dict(), self.save_path)
        

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = APOIS(env, model_type='mlp', save_path='./policy.pth')
    model.learn(lambda_coef=0.1, num_trajectories=10,
               num_online_iterations=10, num_offline_iterations=10)