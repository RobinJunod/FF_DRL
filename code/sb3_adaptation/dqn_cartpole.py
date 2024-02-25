#%%
import os

import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env


from ff_mlp import ForwardForwardMLP, negative_shuffling

#TODO: create the feature extractor here in a MLP

# Example feature extractor architecture // TODO replace with FF MLP
class PreTrainedFeatureExtractor(nn.Module):
    def __init__(self):
        super(PreTrainedFeatureExtractor, self).__init__()
        self.layer1 = nn.Linear(4, 64)  # CartPole has 4 features
        self.layer2 = nn.Linear(64, 4)

    def forward(self, x):
        #x = th.relu(self.layer1(x))
        #x = th.relu(self.layer2(x))
        return x
    
    

# Assuming this model is pre-trained, you'd load its weights here
# feature_extractor = PreTrainedFeatureExtractor()
# feature_extractor.load_state_dict(torch.load("path_to_your_saved_model.pt"))


class CustomSB3FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_extractor_model : ForwardForwardMLP):
        features_dim = feature_extractor_model._dim_ouput
        super(CustomSB3FeatureExtractor, self).__init__(observation_space, features_dim=features_dim)  # Set your features_dim based on your model's output 96
        self.feature_extractor = feature_extractor_model

    def forward(self, observations):
        return self.feature_extractor(observations)
    




#%%



if __name__ == '__main__':
    # Parameters
    train_bp = False
    
    # Creating logs dir and saves
    models_dir = "DQN_cartpole"
    logdir = "logs_cartpole"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Initialize your environment
    env_id = "CartPole-v1"
    env = make_vec_env(env_id, n_envs=1)
    
    #feature_extractor = PreTrainedFeatureExtractor()
    feature_extractor = ForwardForwardMLP(dims=[4, 20, 10, 10])
    
    custom_policy_kwargs = {
        "features_extractor_class": CustomSB3FeatureExtractor,
        "features_extractor_kwargs": {"feature_extractor_model": feature_extractor},
        "net_arch": []  # No hidden layers
    }


    if train_bp: 
        # Initialize the model with the custom feature extractor
        model = DQN("MlpPolicy", 
                    env,
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
        model.learn(total_timesteps=500_000)
        # Evaluate the model
        # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        # print(f"Mean reward: {mean_reward} +/- {std_reward}")
    else:
        # Initialize the model with the custom feature extractor
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
        # infos : obs.shape = [32,4] or [1,4]
        # features : only a tensor (no gradient)
        for fe in range(10): # loop the nb of features extractor to train
            print('Start training feature extractor :', fe)
            # Train the model (the feature extractor's weights will remain frozen if not included in the optimizer)
            model.learn(total_timesteps=200_000)
            
            # get the positive samples from mem buff 
            positive_data = model.replay_buffer.observations.reshape(10000, 4)
            positive_data = th.from_numpy(positive_data)
            positive_data, negative_data = negative_shuffling(positive_data)
            
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
