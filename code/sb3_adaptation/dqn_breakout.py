#%%
import os
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ff_cnn import FeatureExtractorForwardForward, FFConvNet


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

## Creating the custom network
#custom_policy_kwargs = {
#    "features_extractor_class": CustomCNN,
#    "features_extractor_kwargs": {"features_dim": 128},  # Output dimension
#    "net_arch": []  # No hidden layers
#}


custom_policy_kwargs = {
    "features_extractor_class": FFConvNet,
    "features_extractor_kwargs": {"features_dim": 51_904},  # Output dimension
    "net_arch": []  # No hidden layers
}




model = DQN("MlpPolicy",
            env,
            policy_kwargs=custom_policy_kwargs,
            learning_rate=1e-4,
            buffer_size=1_000,
            batch_size=32,
            learning_starts=100,
            target_update_interval=1_000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            optimize_memory_usage=False,
            verbose=0,
            tensorboard_log=logdir)



#%%
def train_feature_extractor(feature_extractor, memory, num_epochs, batch_size):
    """Train the feature extractor (FE) from data store in a memory object
    Args:
        memory (replay_memory.ReplayBuffer): Object from class RepalyMemory filled with data
        num_epochs (int): number of epochs
        batch_size (int): size of the batch
    Returns:
        _type_: _description_
    """
    for epoch in range(num_epochs):
        epoch_loss_mean = 0
        for n in range(int(memory.size/batch_size)):
            x_pos, x_neg = memory.sample_pos_neg_shuffling(n*batch_size, batch_size=batch_size)
            x_pos = th.tensor(x_pos)
            x_neg = th.tensor(x_neg)
            loss = feature_extractor.train(x_pos, x_neg)
            epoch_loss_mean = (n*epoch_loss_mean + loss)/(n+1)
        print (f'Loss mean at epoch {epoch} : {epoch_loss_mean}')
        
    features_size = feature_extractor.respresentation_vects(th.tensor(memory.get_state(0)).unsqueeze(0)).shape[1]
    return feature_extractor, features_size


if __name__ == '__main__':
    n_training = 10
    TIMESTEPS = 2_000

    #for fe in range(n_training):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS}")
        
    # Train the feature extractor #TODO : finds place to modify code
    # model.replay_buffer.observations.shape (1000, 1, 4, 84, 84)
    train_feature_extractor(feature_extractor, 
                            ff_agent.memory, # memory state to train feature extractor
                            batch_size=32, 
                            num_epochs=5) 

# update the model :
    
    