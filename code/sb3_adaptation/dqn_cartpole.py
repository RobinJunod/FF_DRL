#%%
import os

import torch as th
import torch.nn as nn

import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from ff_mlp import ForwardForwardMLP, negative_shuffling

# Assuming this model is pre-trained, you'd load its weights here
# feature_extractor.load_state_dict(torch.load("path_to_your_saved_model.pt"))


class ForwardForwardFE(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_extractor_model : ForwardForwardMLP):
        features_dim = feature_extractor_model._dim_ouput
        super(ForwardForwardFE, self).__init__(observation_space, features_dim=features_dim)  # Set your features_dim based on your model's output 96
        self.feature_extractor = feature_extractor_model
        
    def forward(self, observations):
        return self.feature_extractor(observations)
    



if __name__ == '__main__':
    
    # Creating logs dir and saves
    models_dir = "DQN_cartpole"
    logdir = "logs_cartpole"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Initialize your environment
    env = make_vec_env('CartPole-v1', n_envs=1)
    
    feature_extractor = ForwardForwardMLP(dims=[4, 50, 20, 20])
    
    custom_policy_kwargs = {
        "features_extractor_class": ForwardForwardFE,
        "features_extractor_kwargs": {"feature_extractor_model": feature_extractor},
        "net_arch": []  # No hidden layers
    }


    # Initialize the model with the custom feature extractor
    #model = DQN("MlpPolicy", 
    #            env,
    #            policy_kwargs=custom_policy_kwargs,
    #            learning_rate=1e-4,
    #            buffer_size=10_000,
    #            batch_size=32,
    #            learning_starts=1_000,
    #            target_update_interval=50,
    #            train_freq=4,
    #            gradient_steps=1,
    #            exploration_fraction=0.5,
    #            exploration_final_eps=0.01,
    #            optimize_memory_usage=False,
    #            verbose=0,
    #            tensorboard_log=logdir)
    
    model = DQN("MlpPolicy", 
                env,
                policy_kwargs=custom_policy_kwargs,
                learning_rate=1e-3,
                buffer_size=10_000,
                batch_size=32,
                exploration_initial_eps=1.0,  # Start with pure exploration
                exploration_final_eps=0.1,  # Lower exploration over time
                exploration_fraction=0.1,  # Adjust the fraction of training to reduce epsilon
                optimize_memory_usage=False, 
                verbose=1)
    
    
    # infos : obs.shape = [32,4] or [1,4]
    # features : only a tensor (no gradient)
    for fe in range(10): # loop the nb of features extractor to train
        print('Start training feature extractor :', fe)
        # Train the model (the feature extractor's weights will remain frozen if not included in the optimizer)
        if fe == 0: # train the first time just to fill the mem buffer
            model.learn(total_timesteps=10_000, log_interval=100)
        else: 
            model.learn(total_timesteps=200_000, log_interval=100)
            
        # get the positive samples from mem buff 
        positive_data = model.replay_buffer.observations.reshape(10_000, 4)
        positive_data = th.from_numpy(positive_data)
        positive_data, negative_data = negative_shuffling(positive_data)
        
        feature_extractor.train_ll(positive_data, negative_data, num_epochs=100)
        
        custom_policy_kwargs = {
        "features_extractor_class": ForwardForwardFE,
        "features_extractor_kwargs": {"feature_extractor_model": feature_extractor},
        "net_arch": []  # No hidden layers
        }
        
        model = DQN("MlpPolicy", 
            env,
            policy_kwargs=custom_policy_kwargs,
            learning_rate=1e-3,
            buffer_size=10_000,
            batch_size=32,
            exploration_initial_eps=1.0,  # Start with pure exploration
            exploration_final_eps=0.1,  # Lower exploration over time
            exploration_fraction=0.1,  # Adjust the fraction of training to reduce epsilon
            optimize_memory_usage=False, 
            verbose=1)

