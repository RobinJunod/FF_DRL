#%%
import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ff_cnn import ForwardForwardCNN, negative_data_shuffle, show_image



class CustomSB3FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_extractor_model : ForwardForwardCNN):
        features_dim = feature_extractor_model._dim_output
        super(CustomSB3FeatureExtractor, self).__init__(observation_space, features_dim=features_dim)  # Set your features_dim based on your model's output 96
        self.feature_extractor = feature_extractor_model

    def forward(self, observations):
        return self.feature_extractor(observations)


models_dir = "DQN_breakout"
logdir = "logs_breakout"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_atari_env(
    "BreakoutNoFrameskip-v4",
    n_envs=1,
    seed=42,
)
env = VecFrameStack(env, n_stack=4)
env.reset()

# Create the feature extractor network the forward forward net
feature_extractor = ForwardForwardCNN()
custom_policy_kwargs = {
    "features_extractor_class": CustomSB3FeatureExtractor,
    "features_extractor_kwargs": {"feature_extractor_model": feature_extractor},
    "net_arch": []  # No hidden layers
}


model = DQN("MlpPolicy", 
            env,
            policy_kwargs=custom_policy_kwargs,
            learning_rate=1e-3,
            buffer_size=1_000,
            batch_size=32,
            exploration_initial_eps=1.0,  # Start with pure exploration
            exploration_final_eps=0.1,  # Lower exploration over time
            exploration_fraction=0.1,  # Adjust the fraction of training to reduce epsilon
            optimize_memory_usage=False, 
            verbose=1)


# features : only a tensor (no gradient)
for fe in range(10): # loop the nb of features extractor to train
    print('Start training feature extractor :', fe)
    # Train the model (the feature extractor's weights will remain frozen if not included in the optimizer)
    model.learn(total_timesteps=10_000)
    
    # get the positive samples from mem buff 
    positive_data = np.squeeze(model.replay_buffer.observations, axis=1) # reshape to have ((B x C x H x W))
    positive_data = th.from_numpy(positive_data)
    negative_data = th.from_numpy(negative_data_shuffle(positive_data))
    
    feature_extractor.train_ll(positive_data, negative_data, num_epochs=100)
    
    custom_policy_kwargs = {
    "features_extractor_class": CustomSB3FeatureExtractor,
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
    