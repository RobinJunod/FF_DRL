#%%
import numpy as np
import matplotlib.pyplot as plt

import torch

class ReplayBuffer:
    def __init__(self, max_size=1_000_000, obs_height=84, obs_width=84, stack_size=4):
        self.max_size = max_size
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.stack_size = stack_size

        # Initialize the buffer for storing obs and other data
        self.obs = np.zeros((max_size, obs_height, obs_width), dtype=np.bool_)
        self.action = np.zeros(max_size, dtype=np.int8)
        self.reward = np.zeros(max_size, dtype=np.int8)
        self.done = np.zeros(max_size, dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def remember(self, obs, action, reward, done):
        # Add a obs and corresponding data to the buffer
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_state(self, idx):
        # Get a state by stacking 'stack_size' obs ending at index 'idx'
        idx = (idx + self.size) % self.size  # Handle negative indices
        state = np.zeros((self.stack_size, self.obs_height, self.obs_width), dtype=np.float32)
        for i in range(self.stack_size):
            index = (idx - i) % self.size
            state[self.stack_size - 1 - i] = self.obs[index]
            # Check if the obs at 'index' is the first obs after an episode ended
            if i != 0 and self.done[index]:
                # If 'done' flag is encountered, repeat the same obs for all previous obs in the stack
                state[:self.stack_size - 1 - i] = self.obs[index]
                break
        return state

    def sample(self, batch_size=32):
        # Sample a batch of states, actions, rewards, next states, and dones
        indices = np.random.choice(self.size, batch_size, replace=False)
        #states = np.array([self.get_state(idx) for idx in indices])
        #next_states = np.array([self.get_state(idx + 1) for idx in indices])
        #actions = self.action[indices]
        #rewards = self.reward[indices]
        #dones = self.done[indices]
        states = torch.tensor([self.get_state(idx) for idx in indices], dtype=torch.float32)
        next_states = torch.tensor([self.get_state(idx + 1) for idx in indices], dtype=torch.float32)
        actions = torch.tensor(self.action[indices], dtype=torch.long)
        rewards = torch.tensor(self.reward[indices], dtype=torch.float32)
        dones = torch.tensor(self.done[indices], dtype=torch.float32)
        return states, actions, rewards, next_states, dones
    
    def show_rnd_state(self):
        state,_,_,_,_ = self.sample(batch_size=1)
        state = state.squeeze(0)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            axes[i].imshow(state[i, :, :], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Frame {i + 1}')
        plt.show()
#%%

if __name__ == '__main__':
    import gym
    from env_wrapper import BreakoutWrapper
    
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
    env = BreakoutWrapper(env)
    obs, info = env.reset()
    memory = ReplayBuffer(max_size=1_000)

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        memory.remember(obs, 0, reward, done)
        if done:
            break
    
    memory.show_rnd_state()
    