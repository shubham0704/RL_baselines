import gym
import torch
import numpy as np
import random
from pois import method_factory
from stable_baselines3 import PPO
from sb3_contrib import TRPO
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import BaseCallback

"""
TODO:
Add a callback to the train_kwargs of each method to record the episode returns.
for APOIS and PPOIS
Both of these methods have a num_iterations parameter. 
We should also support a tensorboard logging of the returns.

Ideally, we should have the same interface for all the methods.
"""

class EpisodeReturnCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []
        self.current_episode_return = 0

    def _on_step(self) -> bool:
        self.current_episode_return += self.locals["rewards"][0]
        
        if self.locals["dones"][0]:
            self.episode_returns.append(self.current_episode_return)
            self.logger.record("episode_return", self.current_episode_return)
            self.current_episode_return = 0
        
        return True

method_factory["trpo"] = TRPO
method_factory["ppo"] = PPO

evaluate_config = {
    "linear": {
        "seeds": [10, 109, 904, 160, 570],
        "method":{
            "p-pois": {
                "init_kwargs": {
                    "policy": "linear"
                },
                "train_kwargs": {
                    "num_offline_iterations": 10,
                    "num_iterations": 5,
                    "episodes_per_iteration": 10
                }
            },
            "a-pois": {
                "init_kwargs": {
                    "policy": "linear"
                },
                "train_kwargs": {
                    "num_offline_iterations": 10,
                    "num_iterations": 5,
                    "episodes_per_iteration": 10
                }
            },
            "trpo": {
                "init_kwargs": {
                    "policy": "MlpPolicy",
                    "policy_kwargs": {
                        "net_arch": []
                    }
                },
                "train_kwargs": {
                    "total_timesteps": 50,
                    "callback": EpisodeReturnCallback()
                }
            },
            "ppo": {
                "init_kwargs": {
                    "policy": "MlpPolicy",
                    "policy_kwargs": {
                        "net_arch": []
                    },
                    "n_epochs": 10
                },
                "train_kwargs": {
                    "total_timesteps": 50,
                    "callback": EpisodeReturnCallback(),
                }
            }
        }
        
    },
    "mlp": {
        "seeds": [10, 109, 904, 160, 570],
        "method":{
            "p-pois": {
                "init_kwargs": {
                    "policy": "MlpPolicy",
                },
                "train_kwargs": {
                    "num_offline_iterations": 20,
                    "num_iterations": 5,
                    "episodes_per_iteration": 10
                }
            },
            "a-pois": {
                "init_kwargs": {
                    "policy": "MlpPolicy",
                },
                "train_kwargs": {
                    "num_offline_iterations": 20,
                    "num_iterations": 5,
                    "episodes_per_iteration": 10
                }
            },
            "trpo": {
                "init_kwargs": {
                    "policy": "MlpPolicy",
                    "policy_kwargs": {
                        "net_arch": [100, 50, 25]
                    }
                },
                "train_kwargs": {
                    "total_timesteps": 50,
                    "callback": EpisodeReturnCallback(),
                }
            },
            "ppo": {
                "init_kwargs": {
                    "policy": "MlpPolicy",
                    "policy_kwargs": {
                        "net_arch": [100, 50, 25]
                    },
                    "n_epochs": 10
                },
                "train_kwargs": {
                    "total_timesteps": 50,
                    "callback": EpisodeReturnCallback(),
                }
                }
            }
    }
}

env_config = {
    "CartPole-v1": {
       "delta":{
           "p-pois": 
               {"lambda_coef": 0.4},
           "a-pois": 
               {"lambda_coef": 0.4},
           "trpo": 
               {"learning_rate": 0.1},
           "ppo": 
               {"learning_rate": 0.01}
       }
    }
}
returns = {}
methods_to_evaluate = ['ppo']
# running this for cartpole-v1
deltas = env_config["CartPole-v1"]["delta"]
for model_type, config in evaluate_config.items():
    returns[model_type] = {}
    avg_returns = []
    for method_name in methods_to_evaluate:
        print(f"Running {method_name} for {model_type}")
        method_config = config["method"][method_name]
        for seed in tqdm(config["seeds"]):
            # set seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)   
            env = gym.make("CartPole-v1")
            init_kwargs = method_config["init_kwargs"]
            additional_kwargs = deltas[method_name]
            model = method_factory[method_name](env=env, 
                                                **init_kwargs, **additional_kwargs)
            if method_name == "trpo" or method_name == "ppo":
                callback = method_config["train_kwargs"]["callback"]
                model.learn(**method_config["train_kwargs"])
                return_vals = callback.episode_returns
            else:
                return_vals = model.learn(**method_config["train_kwargs"])
            avg_returns.append(return_vals)
        returns[model_type][method_name] = (np.mean(avg_returns, axis=0), 
                        np.std(avg_returns, axis=0))
            
# plot the results of returns
# Plotting results

plt.figure()
for model_type, config in evaluate_config.items():
    for method_name in methods_to_evaluate:
        print(f"Plotting {method_name} for {model_type}")
        print("returns shape: ", returns[model_type][method_name][0].shape)
        mean_reward, std_reward = returns[model_type][method_name]
        plt.plot(mean_reward, label=method_name)
        plt.fill_between(np.arange(mean_reward.shape[0]), mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
plt.title(f"CartPole-v1 - Average Return")
plt.xlabel("Episodes")
plt.ylabel("Average Return")
plt.legend()
plt.savefig("cartpole_v1_returns.png")
plt.show()

