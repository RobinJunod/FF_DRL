#%%
import os
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ff_cnn import ForwardForwardCNN, negative_data_gen




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



feature_extractor = ForwardForwardCNN()

custom_policy_kwargs = {
    "features_extractor_class": CustomSB3FeatureExtractor,
    "features_extractor_kwargs": {"feature_extractor_model": feature_extractor},
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




# features : only a tensor (no gradient)
for fe in range(10): # loop the nb of features extractor to train
    print('Start training feature extractor :', fe)
    # Train the model (the feature extractor's weights will remain frozen if not included in the optimizer)
    model.learn(total_timesteps=200_000)
    
    # get the positive samples from mem buff 
    positive_data = model.replay_buffer.observations.reshape(10000, 4)
    positive_data = th.from_numpy(positive_data)
    positive_data, negative_data = negative_data_gen(positive_data)
    
    feature_extractor.train_ff(positive_data, negative_data, num_epochs=300)
    
    custom_policy_kwargs = {
    "features_extractor_class": CustomSB3FeatureExtractor,
    "features_extractor_kwargs": {"feature_extractor_model": feature_extractor},
    "net_arch": []  # No hidden layers
    }
    
    model = DQN("MlpPolicy", 
                env,
                policy_kwargs=custom_policy_kwargs,
                learning_rate=1e-4,
                buffer_size=10_000,
                batch_size=32,
                learning_starts=1_000,
                target_update_interval=1_000,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.5,
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
    
    