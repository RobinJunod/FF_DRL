# %%
import gym
import ale_py
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

from collections import deque

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136 , 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.q_network = QNetwork(action_size)
        self.target_q_network = QNetwork(action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)
        self.transforms = T.Compose([T.ToPILImage(), T.Resize((84, 84)), T.Grayscale(), T.ToTensor()])
        
    def stack_frames(self, frames):
        # Stack the consecutive frames as a single 4-channel image
        return torch.cat(frames, dim=0)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, stacked_state):
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            stacked_state_ = stacked_state.unsqueeze(0)
            q_values = self.q_network(stacked_state_)
            return np.argmax(q_values.cpu().data.numpy())

    def train(self):
        # Here states and next_states are 4 succ img in grey scale
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        dones = torch.tensor(dones, dtype=torch.float32)
        # get all the Q values at each states 
        q_values = self.q_network(states)
        # The next q values are determined by the more stable network (off policy)
        next_q_values = self.target_q_network(next_states).max(1).values
        # The target q value is the computed as the sum of the reward and the futur best Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Select the Q values of the actions taken
        q_values = q_values.gather(1, actions.view(-1, 1))
        # compute the loss from the q values differences (the prediction and the target) (2-arrays of length 32 (batch-size))
        loss = F.smooth_l1_loss(q_values, target_q_values.view(-1, 1))
        # loss_mean = ((episode_length-1)*loss_mean + loss.item())/episode_length
        # Optimization using basic pytorch code
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Decrease the epsilon 
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


if __name__ == '__main__':
    # Initialize the Breakout environment
    env = gym.make('Breakout-v0', render_mode='human')
    state_size = (1, 84, 4 * 84)  # Four consecutive frames as input
    action_size = env.action_space.n

    # Initialize the DQN agent
    agent = DQNAgent(state_size, action_size)

    # Training loop
    episodes = 300
    for episode in range(episodes):
        print('Start episode :', episode)
        state, infos = env.reset()
        state = agent.transforms(state)
        frames = [state] * 4  # Initialize the stack with the same frame
        total_reward = 0
        done = False
        terminated = False
        step = 0

        while not done and not terminated:#  and step < 300:
            step += 1
            #print('state :', state.shape)
            stacked_state = agent.stack_frames(frames)  # Stack the frames            
            action = agent.act(stacked_state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = agent.transforms(next_state)
            frames.pop(0)  # Remove the oldest frame
            frames.append(next_state)  # Add the newest frame
            
            stacked_next_state = agent.stack_frames(frames) # Create stacked next frame
                        
            agent.remember(stacked_state, action, reward, stacked_next_state, done)
            # print(f'stacked state{stacked_state.shape}, action{action}, reward{reward}, next {stacked_next_state.shape}, done{done}')
            if done:
                agent.update_target_q_network()
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
                
            agent.train()
            total_reward += reward
            # env.render()
            
        print('Total Reward over the episode:', total_reward)

    env.close()
    torch.save(agent.q_network.state_dict(), 'dqn_breakout_q_network.pth')


#%%
print('patae')
