# %%
import gym
import time

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import deque
from  model import QNetwork
from atari_wrappers import wrap_deepmind

from replay_buffer import ReplayBuffer

# Define the DQN Agent
class DQNAgent:
    def __init__(self, action_size=3):
        self.action_size = action_size
        # Hyperparameters
        self.memory = ReplayBuffer(1000000, 4) # 1'000'000 in paper, rep mem size
        self.batch_size = 32 
        self.gamma = 0.99
        self.target_upadte_freq = 10000
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.final_exploration_step = 1000000 # 1'000'000 in paper, number step to stop exploring
        self.epsilon_decay = (1-0.1)/self.final_exploration_step # Linear decay
        self.epsilon_decay_exp = self.epsilon_min**(1/self.final_exploration_step ) # Exponentional decay
        # Deep Neural Network
        self.q_network = QNetwork(action_size)
        self.target_q_network = QNetwork(action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)
        # Loggers



    def remember(self, state, action, reward, next_state, done):
        # Append (s,a,r,s') to replay memory
        self.memory.append((state, action, reward, next_state, done))

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
    def act_egreedy(self, state):
        """Select an action based on the Epsioln-greedy method
        Args:
            state (tensor 4x84x84): The input of the network (4 img preproc)
        Returns:
            action: The action that must be played
        """
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            return self.act(state)
        
    def act(self, state):
        # Choice of the agent
        state = state.unsqueeze(0)
        q_values = self.q_network(state)
        action = torch.argmax(q_values).item()
        return action
    
    def optimize_model(self):
        # Here states and next_states are 4 succ img in grey scale
        if len(self.memory) < self.batch_size:
            return
        # Get samples from replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        dones = torch.tensor(dones, dtype=torch.int)
        
        # Get all the Q values at each states 
        q_values = self.q_network(states)
        # The next Q values are determined by the more stable network (off policy)
        next_q_values = self.target_q_network(next_states).max(1).values
        # The target q value is the computed as the sum of the reward and the futur best Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Select the Q values of the actions taken
        q_values = q_values.gather(1, actions.view(-1, 1))
        # Compute the loss from the q values differences (the prediction and the target) (2-arrays of length 32 (batch-size))
        loss = F.smooth_l1_loss(q_values, target_q_values.view(-1, 1))
        # Optimization using basic pytorch code
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Decrease the epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay) # lin decay
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_exp) # exp decay
        return loss.item()
        
#%
def train(agent, env, nb_epsiode=100, save_model = True, render=False):
    
    t_steps = 0
    
    for episode in range(nb_epsiode):
        print('Start episode :', episode)
        state, info = env.reset()


        total_reward = 0
        done = False
        step = 0
        while not done:
            step += 1
            t_steps += 1
            #print('state :', state.shape)
            action = agent.act_egreedy(state) # Agent selects action
            
            next_state, reward, terminated, truncated, info = env.step(action)
            if render:
                env.render()
            done = terminated or truncated
            total_reward += reward

            # print(f'stacked state{stacked_state.shape}, action{action}, reward{reward}, next {stacked_next_state.shape}, done{done}')
            agent.remember(state, action, reward, next_state, done)
            loss = agent.optimize_model()
            
            if t_steps % 30000 == 0:
                env.save_current_state_images('img_state', t_steps)
                print(f'loss at ts {t_steps} : {loss}')
            
            if t_steps % agent.target_upadte_freq == 0:
                agent.update_target_q_network()
                print(f"Update Target Net : Episode: {episode + 1}, Epsilon: {agent.epsilon}")
            
            
            state = next_state
            
        print(f'Total Reward over the episode:{total_reward} , Current epsilon : {agent.epsilon}')
 
    env.close()
    if save_model:
        torch.save(agent.q_network.state_dict(), 'dqn_breakout_q_network.pth')
    

def test(agent, pth_path, env, save_video=False, render=True):
    # Load the trained model
    agent.q_network.load_state_dict(torch.load(pth_path))
    agent.q_network.eval()
    t_steps = 0

    for episode in range(10):  # You can adjust the number of episodes for testing
        print('Start testing episode:', episode)
        state, info = env.reset()

        total_reward = 0
        done = False
        step = 0

        while not done:
            step += 1
            t_steps += 1

            action = agent.act_egreedy(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f'Total Reward for testing episode {episode}: {total_reward}')

    env.close()
#%%
if __name__ == '__main__':
    # Initialize the Breakout environment
    env = gym.make('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env)
    
    
    #%%
    action_size = 3

    # Initialize the DQN agent
    agent = DQNAgent(action_size)

    # Train DQL
    train(agent,env, nb_epsiode=30000, render=False)


# %%
