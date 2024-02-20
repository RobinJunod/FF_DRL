#%%
import os
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ff_model import FFConvNet


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        #print(observations.shape[0])
        #print(th.arange(observations.shape[0]*128, dtype=th.float).view(observations.shape[0], -1).shape)# test for cst function
        #print(self.linear(self.cnn(observations)).shape)
        return th.arange(observations.shape[0]*128, dtype=th.float).view(observations.shape[0], -1)

# Creating the custom network
custom_policy_kwargs = {
    "features_extractor_class": CustomCNN,
    "features_extractor_kwargs": {"features_dim": 128},  # Output dimension
    "net_arch": []  # No hidden layers
}

models_dir = "DQN"
logdir = "logs"
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

model = DQN("MlpPolicy",
            env,
            policy_kwargs=custom_policy_kwargs,
            learning_rate=1e-4,
            buffer_size=10_000,
            batch_size=32,
            learning_starts=10_000,
            target_update_interval=1_000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            optimize_memory_usage=False,
            verbose=0,
            tensorboard_log=logdir)




#%% Traing part of the network
TIMESTEPS = 10_000
iters=0
#for i in range(100):
iters += 1
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
model.save(f"{models_dir}/{TIMESTEPS*iters}")
#model.replay_buffer.observations # to get obs
#model.replay_buffer.actions # to get actions

