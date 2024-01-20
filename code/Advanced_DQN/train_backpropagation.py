# %%
import gym
import time

import random
import numpy as np
import pandas as pd
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
        self.memory = ReplayBuffer(max_size=500_000, stack_size=4) # 1'000'000 in paper, rep mem size
        self.batch_size = 32
        self.gamma = 0.99
        self.target_update_freq = 7_000
        self.epsilon = 0.5
        self.epsilon_min = 0.02
        self.final_exploration_step = 500_000 # 1'000'000 in paper, number step to stop exploring
        self.epsilon_decay = (1-0.1)/self.final_exploration_step # Linear decay
        self.epsilon_decay_exp = self.epsilon_min**(1/self.final_exploration_step ) # Exponentional decay
        self.no_learning_steps = 100_000
        # Deep Neural Network
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to use gpu
        
        self.q_network = QNetwork(self.action_size)
        self.target_q_network = QNetwork(self.action_size)
        # TODO : remove if want to train from start
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001, eps=1.5e-4)
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
        with torch.no_grad():
            # Has to Convert to (N, C, H, W), only format handled by conv2d        
            state = state.unsqueeze(0)
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
            return action
    
    def optimize_model(self):
        # Here states and next_states are 4 succ img in grey scale
        if self.memory.size < self.no_learning_steps:
            return
        # Get batch sample from memory buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)
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
        # loss = F.mse_loss(q_values, target_q_values.view(-1, 1))
        # loss = F.l1_loss(q_values, target_q_values.view(-1, 1))
        # Optimization using basic pytorch code
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        # Decrease the epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay) # lin decay
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_exp) # exp decay
        return loss.item()

def train(agent, env, nb_epsiode=100, save_model=True):
    
    t_steps = 0
    max_rew = 0 
    logs = pd.DataFrame(columns=['rew', 'max', 'ep'])
    for episode in range(nb_epsiode):
        print('Start episode :', episode)
        obs, _ = env.reset()
        # Add the first state to the replay memory TODO : verify if good
        agent.memory.remember(obs, 0, 0, True) # a=No-op, r=0, done=False
        total_reward = 0
        done = False
        step = 0
        while not done:
            step += 1
            t_steps += 1

            # Extract the state from the replay memory buffer (last obs in memory)
            # Debugging .ptr-1 is crucial
            state = torch.tensor(agent.memory.get_state(agent.memory.ptr - 1), dtype=torch.float32)
            action = agent.act_egreedy(state) # Agent selects action
            
            next_obs, reward, done, _, reward_unclipped = env.step(action)
            
            total_reward += reward_unclipped # For plotting purpuses
            # Adding s a r d to memory buffer #TODO:verify if good next_obs - obs
            agent.memory.remember(next_obs, action, reward, done)
            loss = agent.optimize_model()
            
            # obs = next_obs TODO: verify if good
            # update target network        
            if t_steps % agent.target_update_freq == 0:
                agent.update_target_q_network()
                print(f"Update Target Net : Episode: {episode + 1}, Epsilon: {agent.epsilon}")
                print(f'Loss at ts {t_steps} : {loss}')

            # save network weights
            if save_model==True and t_steps % 200_000 == 0:
                model_save_path = f'dqn_breakout_q_network_{t_steps}.pth'
                torch.save(agent.q_network.state_dict(), model_save_path)
            
                
        if total_reward > max_rew:
            max_rew = total_reward
        #agent.memory.show_state()
        print(f'Reward episode:{total_reward}, max rew :{max_rew} ,  epsilon : {agent.epsilon}')
        if episode%100 == 0:
            new_row = { 'episode' : [episode],
                        'rew': [total_reward], 
                        'max': [max_rew], 
                        'ep': [agent.epsilon]}
            logs.loc[len(logs)] = new_row
            logs.to_csv('logs.csv', index=False)

    if save_model:
        torch.save(agent.q_network.state_dict(), 'dqn_breakout_q_network.pth')
    env.close()
    

def test(agent, pth_path, env, save_video=False, render=True):
    # Load the trained model
    agent.q_network.load_state_dict(torch.load(pth_path))
    agent.q_network.eval()
    t_steps = 0
    for episode in range(10):  # You can adjust the number of episodes for testing
        print('Start testing episode:', episode)
        obs, _ = env.reset()
        agent.memory.remember(obs, 0, 0, True)
        total_reward = 0
        done = False
        step = 0
        while not done:
            step += 1
            t_steps += 1
            state = torch.tensor(agent.memory.get_state(agent.memory.ptr - 1), dtype=torch.float32)
            action = agent.act(state)
            next_state, reward, done, _, reward_unclipped = env.step(action)
            total_reward += reward_unclipped
            agent.memory.remember(next_state, reward, action, done)
            if render:
                env.render()
                
        print(f'Total Reward for testing episode {episode}: {total_reward}')
    env.close()

#%%   
if __name__ == '__main__':
    
    # Initialize the Breakout environment
    #env = gym.make('BreakoutDeterministic-v4')
    #env = gym.make('ALE/Breakout-v5')
    env = gym.make('BreakoutNoFrameskip-v4')
    env = BreakoutWrapper(env) 
    
    # Initialize the DQN agent
    agent = DQNAgent()
    # Train DQL
    train(agent,env, nb_epsiode=20_000, save_model=True)

     #%% Cell to run the test mode / inference
    env2 = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    env2 = BreakoutWrapper(env2)
    agent2 = DQNAgent()
    pth_path = 'model_weight_pth/dqn_breakout_q_network_insanelygood.pth'
    test(agent2, pth_path, env2, save_video=False, render=True)