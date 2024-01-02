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

from model import QNetwork
from env_wrapper import BreakoutWrapper
from replay_memory import ReplayBuffer

# Define the DQN Agent
class DQNAgent:
    def __init__(self):
        self.action_size = 3
        # Hyperparameters
        self.memory = ReplayBuffer(max_size=100_000, stack_size=4) # 1'000'000 in paper, rep mem size
        self.batch_size = 32
        self.gamma = 0.99
        self.target_update_freq = 1_000
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.final_exploration_step = 1_000_000 # 1'000'000 in paper, number step to stop exploring
        self.epsilon_decay = (1-0.1)/self.final_exploration_step # Linear decay
        self.epsilon_decay_exp = self.epsilon_min**(1/self.final_exploration_step ) # Exponentional decay
        self.no_learning_steps = 10_000
        # Deep Neural Network
        self.q_network = QNetwork(self.action_size).to(self.device)
        self.target_q_network = QNetwork(self.action_size).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        # Loggers


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
        # Has to Convert to (N, C, H, W), only format handled by conv2d        
        state = state.unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        action = torch.argmax(q_values).item()
        return action
    
    def optimize_model(self):
        # Here states and next_states are 4 succ img in grey scale
        if self.memory.size < self.no_learning_steps:
            return
        # Get batch sample from memory buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=32)
        # Get all the Q values at each states 
        q_values = self.q_network(states)
        # The next Q values are determined by the more stable network (off policy)
        next_q_values = self.target_q_network(next_states).max(1).values
        # The target q value is the computed as the sum of the reward and the futur best Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Select the Q values of the actions taken
        q_values = q_values.gather(1, actions.view(-1, 1))
        # Compute the loss from the q values differences (the prediction and the target) (2-arrays of length 32 (batch-size))
        loss = F.smooth_l1_loss(q_values, target_q_values.view(-1, 1)).to(self.device)

        # Optimization using basic pytorch code
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Decrease the epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay) # lin decay
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_exp) # exp decay
        return loss.item()

def train(agent, env, nb_epsiode=100, save_model=True, render=False):
    
    t_steps = 0
    for episode in range(nb_epsiode):
        print('Start episode :', episode)
        obs, _ = env.reset()
        # Add the first state to the replay memory
        agent.memory.remember(obs, 0, 0, True) # a=No-op, r=0, done=False
        total_reward = 0
        done = False
        step = 0
        while not done:
            step += 1
            t_steps += 1

            # Extract the state from the replay memory buffer (last obs in memory)
            state = torch.tensor(agent.memory.get_state(agent.memory.ptr), dtype=torch.float32)
            action = agent.act_egreedy(state) # Agent selects action
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # print(f'stacked state{stacked_state.shape}, action{action}, reward{reward}, next {stacked_next_state.shape}, done{done}')
            agent.memory.remember(obs, action, reward, done)
            
            loss = agent.optimize_model()
            
            obs = next_obs
            # update target network        
            if t_steps % agent.target_update_freq == 0:
                agent.update_target_q_network()
                print(f"Update Target Net : Episode: {episode + 1}, Epsilon: {agent.epsilon}")
                print(f'Loss at ts {t_steps} : {loss}')
            # save network weights
            if save_model==True and t_steps % 200_000 == 0:
                model_save_path = f'dqn_breakout_q_network_{t_steps}.pth'
                torch.save(agent.q_network.state_dict(), model_save_path)
            
            
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
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            step += 1
            t_steps += 1
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            done = terminated or truncated
            total_reward += reward
            state = next_state
        print(f'Total Reward for testing episode {episode}: {total_reward}')
    env.close()
    
if __name__ == '__main__':
    
    # Initialize the Breakout environment
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    env = BreakoutWrapper(env)
     
    # Initialize the DQN agent
    agent = DQNAgent()
    # Train DQL
    train(agent,env, nb_epsiode=30_000, render=False)

    #%% To see the evolution of the states from 1 to ... n
    done = agent.memory.done
    done_idx = np.where(done==True)
    # Interupt the simulation
    agent.memory.show_state(state=agent.memory.get_state(1))