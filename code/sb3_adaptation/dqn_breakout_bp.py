#%%
import torch as th
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class CustomCNN(nn.Module):
    def __init__(self, n_input_channels: int = 4):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=10, stride=6)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.conv3])
        self.outputlayer = nn.Linear(51904, 4)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = th.relu(layer(x))
            outputs.append(x)
        flattened_tensors = [tensor.flatten(start_dim=1) for tensor in outputs]
        output = self.outputlayer(th.cat(flattened_tensors, dim=1))
        return output
        
env = make_atari_env(
    "BreakoutNoFrameskip-v4",
    n_envs=1,
    seed=42,
)
env = VecFrameStack(env, n_stack=4)
env.reset()

# Train the agent
model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=32,
        learning_starts=1_000,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
        verbose=1)


#%%
model.q_net.features_extractor = Identity()
model.q_net.q_net = CustomCNN()
model.q_net_target.features_extractor = Identity()
model.q_net_target.q_net = CustomCNN()
model.policy.optimizer = optim.Adam(model.q_net.q_net.parameters(), lr=model.learning_rate)

#%%
model.learn(total_timesteps=100_000)

# %%
