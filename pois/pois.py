from learn import train_p_pois_with_line_search, train_a_pois_with_line_search
from models import model_factory
import gym
import torch
import os



class POIS:
    def __init__(self, env, model_type='mlp', save_dir='./artifacts', pois_type='a-pois'):
        self.env = env
        self.model_type = model_type
        self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box): # continuous action space
            self.action_dim = env.action_space.shape[0] 
        else: # discrete action space
            self.action_dim = env.action_space.n
        self.policy_old = model_factory[model_type](self.state_dim, self.action_dim)
        self.policy_new = model_factory[model_type](self.state_dim, self.action_dim)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, pois_type+'_'+model_type+".pth")
        self.pois_type = pois_type
    def learn(self, lambda_coef, 
              num_iterations=10, episodes_per_iteration=10, num_offline_iterations=10):
        if self.pois_type == 'a-pois':
            self.policy_new = train_a_pois_with_line_search(self.env, 
                                                        self.policy_old, 
                                                        self.policy_new, 
                                                        lambda_coef=lambda_coef,
                                                        num_iterations=num_iterations,
                                                        episodes_per_iteration=episodes_per_iteration,
                                                        num_offline_iterations=num_offline_iterations
                                                       )
        else:
            self.policy_new = train_p_pois_with_line_search(self.env, 
                                                        self.policy_old, 
                                                        self.policy_new, 
                                                        lambda_coef=lambda_coef,
                                                        num_iterations=num_iterations,
                                                        episodes_per_iteration=episodes_per_iteration,
                                                        num_offline_iterations=num_offline_iterations)
        # save the new policy
        torch.save(self.policy_new.state_dict(), self.save_path)
        

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = POIS(env, model_type='mlp', save_dir='./artifacts', pois_type='p-pois')
    model.learn(lambda_coef=0.1, 
                num_iterations=10, 
                episodes_per_iteration=10, 
                num_offline_iterations=10)  