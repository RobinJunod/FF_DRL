import torch as th
import torch.nn as nn
import torch.optim as optim

from typing import List
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

class CustomMLP(nn.Module):
    def __init__(self, net_arch: List[int] = [4, 50, 20, 20]) -> None:
        super(CustomMLP, self).__init__()
        # Define layers individually
        self.linear1 = nn.Linear(net_arch[0], net_arch[1])
        self.linear2 = nn.Linear(net_arch[1], net_arch[2])
        self.linear3 = nn.Linear(net_arch[2], net_arch[3])
        self.relu = nn.ReLU()
        self.outputlayer = nn.Linear(sum(net_arch[1:]), 2)

    def forward(self, obs: th.Tensor):
        x1 = self.relu(self.linear1(obs))
        x2 = self.relu(self.linear2(x1))
        x3 = self.relu(self.linear3(x2))
        x = th.cat((x1, x2, x3), dim=1)
        output = self.outputlayer(x)
        return output


# Custom MLP Policy
net_arch = [4, 50, 20, 20]  # Custom network architecture




# Make the environment
env = make_vec_env("CartPole-v1")

# Train the agent
model = DQN("MlpPolicy", 
            env,
            learning_rate=1e-3,
            buffer_size=10_000,
            batch_size=32,
            exploration_initial_eps=1.0,  # Start with pure exploration
            exploration_final_eps=0.1,  # Lower exploration over time
            exploration_fraction=0.1,  # Adjust the fraction of training to reduce epsilon
            optimize_memory_usage=False, 
            verbose=1)

model.q_net.q_net = CustomMLP()
model.q_net_target.q_net = CustomMLP()
model.policy.optimizer = optim.Adam(model.q_net.q_net.parameters(), lr=0.001)

#%%
model.learn(total_timesteps=100_000)

# Save the model
model.save("dqn_cartpole_custom_policy")
