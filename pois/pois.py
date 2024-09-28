from .learn import train_p_pois_with_line_search, train_a_pois_with_line_search
from .models import model_factory
import gym
import torch
import os



class POIS:
    def __init__(self, env, policy, save_dir='./artifacts', method='a-pois', lambda_coef=0.01, device='cpu'):
        self.env = env
        self.policy = policy
        self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box): # continuous action space
            self.action_dim = env.action_space.shape[0] 
        else: # discrete action space
            self.action_dim = env.action_space.n
        self.policy_old = model_factory[policy](self.state_dim, self.action_dim, method=method)
        self.policy_new = model_factory[policy](self.state_dim, self.action_dim, method=method)
        self.policy_old.to(device)
        self.policy_new.to(device)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, method+'_'+policy+".pth")
        self.method = method
        self.lambda_coef = lambda_coef
    def learn(self,
              num_iterations=10, 
              episodes_per_iteration=10, 
              num_offline_iterations=10):
        if self.method == 'a-pois':
            self.policy_new, return_vals = train_a_pois_with_line_search(self.env, 
                                                        self.policy_old, 
                                                        self.policy_new, 
                                                        lambda_coef=self.lambda_coef,
                                                        num_iterations=num_iterations,
                                                        episodes_per_iteration=episodes_per_iteration,
                                                        num_offline_iterations=num_offline_iterations
                                                       )
        else:
            self.policy_new, return_vals = train_p_pois_with_line_search(self.env, 
                                                        self.policy_old, 
                                                        self.policy_new, 
                                                        lambda_coef=self.lambda_coef,
                                                        num_iterations=num_iterations,
                                                        episodes_per_iteration=episodes_per_iteration,
                                                        num_offline_iterations=num_offline_iterations)
        # save the new policy
        torch.save(self.policy_new.state_dict(), self.save_path)
        return return_vals
    
    def load_policy(self, path):
        self.policy_new.load_state_dict(torch.load(path))
    
    def predict(self, observations, state=None, episode_start=None, deterministic=True):
        return self.policy_new.predict(observations, state, episode_start, deterministic)
    

APOIS = lambda env, policy, **kwargs: POIS(env, policy, method='a-pois', **kwargs)
PPOIS = lambda env, policy, **kwargs: POIS(env, policy, method='p-pois', **kwargs)

method_factory = {
    'a-pois': APOIS,
    'p-pois': PPOIS
}


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = POIS(env, policy='linear', save_dir='./artifacts', method='p-pois', lambda_coef=0.01)
    return_vals = model.learn( 
                                num_iterations=10, 
                                episodes_per_iteration=10, 
                                num_offline_iterations=10)  
    print(return_vals)