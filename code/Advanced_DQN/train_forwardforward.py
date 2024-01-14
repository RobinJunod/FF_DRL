
# The feature extractor weights should then be fixed,
# and a trainable linear layer is added on top of it
# The new model (extractor+linear layer) is trained as a standard DQN
# Experience (i.e. positive data) is generated with the DQN and used to generate new negative data
# Then the feature extractor is trained with these data
# Repeat 1-4

#%%
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import gym
from gym import wrappers
from gym.utils.save_video import save_video

from replay_memory import ReplayBuffer
from FF_CNN import FFConvNet
from env_wrapper import BreakoutWrapper



def init_feature_extractor(env,
                           batch_size=32, 
                           num_epochs=500, 
                           max_mem_size=100_000
                            ):
    """2 phases : 1 collection random sample over random actions, 2 train feature extractor based on these
    Args:
        env (): gym env with wrapper
        batch_size (int, optional): size of batch. Defaults to 32.
        num_epochs (int, optional): number of epochs to trian FE. Defaults to 500.
        max_mem_size (int, optional): Size of memory buffer. Defaults to 100_000.
    Returns:
        feature extractor(FFConvNet): after training
    """
    feature_extractor = FFConvNet()
    memory = ReplayBuffer()
    env.reset()
    while memory.size < max_mem_size:
        env.reset()
        done = False
        while not done:
            # Play the selected action
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Just store the state action pair
            memory.remember(obs, action, reward, done)
            
    return train_feature_extractor(feature_extractor, memory, num_epochs, batch_size)
    
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
            x_pos = torch.tensor(x_pos)
            x_neg = torch.tensor(x_neg)
            loss = feature_extractor.train(x_pos, x_neg)
            epoch_loss_mean = (n*epoch_loss_mean + loss)/(n+1)
        print (f'Loss mean at epoch {epoch} : {epoch_loss_mean}')
        
    features_size = feature_extractor.respresentation_vects(torch.tensor(memory.get_state(0)).unsqueeze(0)).shape[1]
    return feature_extractor, features_size

class FFAgent:
    def __init__(self, feature_extractor, feature_size,
                memory_max_size=100_000,
                batch_size=32,
                gamma=0.99,
                target_update_freq=3_000,
                epsilon=0.5,
                epsilon_min=0.01,
                epsilon_decay = 'lin', # can take values 'lin' or 'exp'
                final_exploration_step=100_000,
                no_learning_steps=10_000,
                ):
        """This is the calss of the agent, it can update its neural network based on
        what is inside its replay memory. This separates the agent from the environemnt.
        The agent only sees what's inside its memory and learns from it. The network 'brain'
        of the agent is made of 1 Feature extractor and 1 regression layer that need to be trained.
        Args:
            feature_extractor (object ): A fully trained feature extractor
            feature_size (int): The size of the features
        """
        self.action_size = 3
        # Hyperparameters
        self.memory = ReplayBuffer(max_size=memory_max_size, stack_size=4)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.final_exploration_step = final_exploration_step
        self.no_learning_steps = no_learning_steps
        self.epsilon_decay = epsilon_decay.upper()
        self.epsilon_decay_lin = (1-0.1)/self.final_exploration_step # Linear decay
        self.epsilon_decay_exp = self.epsilon_min**(1/self.final_exploration_step ) # Exponentional decay
        # Deep Neural Network
        self.feature_extractor = feature_extractor
        self.regression_layer = nn.Linear(feature_size, env.action_space.n)
        # Target init
        self.target_regression_layer = nn.Linear(feature_size, env.action_space.n)
        self.target_regression_layer.load_state_dict(self.regression_layer.state_dict())
        self.target_regression_layer.eval()
        self.optimizer = optim.Adam(self.regression_layer.parameters(), lr=1e-4)


    def update_target(self):
        self.target_regression_layer.load_state_dict(self.regression_layer.state_dict())
        
    def act_egreedy(self, state):
        """Select an action based on the Epsioln-greedy method
        Args:
            state (torch.tensor 4x84x84): The input of the network (4 img preproc)
        Returns:
            action: The action that must be played
        """
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            # Has to Convert to (N, C, H, W), only format handled by conv2d 
            state = state.unsqueeze(0) 
            feature_state = self.feature_extractor.respresentation_vects(state)
            return self.act(feature_state)
        
    def act(self, feature_state):
        with torch.no_grad():
            q_values = self.regression_layer(feature_state)
            action = torch.argmax(q_values).item()
            return action
    
    def optimize_model(self):
        """DQN algorithm to make the agent learn from its memory
        Returns:
            Loss : return the current loss (only for log purpuses)
        """
        # Here states and next_states are 4 succ img in grey scale
        if self.memory.size < self.no_learning_steps:
            return
        # Get batch sample from memory buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=32)
        features_states = self.feature_extractor.respresentation_vects(states)
        features_next_states = self.feature_extractor.respresentation_vects(next_states)
        # Get all the Q values at each states 
        q_values = self.regression_layer(features_states)
        # The next Q values are determined by the more stable network (off policy)
        next_q_values = self.target_regression_layer(features_next_states).max(1).values
        # The target q value is the computed as the sum of the reward and the futur best Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Select the Q values of the actions taken
        q_values = q_values.gather(1, actions.view(-1, 1))
        # Compute the loss from the q values differences (the prediction and the target) (2-arrays of length 32 (batch-size))
        loss = F.smooth_l1_loss(q_values, target_q_values.view(-1, 1))
        # Optimization using basic pytorch code
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.regression_layer.parameters(), max_norm=1.0)
        self.optimizer.step()
        # Decrease the epsilon
        if self.epsilon_decay == 'LIN':
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay) # lin decay
        elif self.epsilon_decay == 'EXP':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_exp) # exp decay
        return loss.item() # Not mendatory

def train_dqn_lastlayer(agent, env, feature_extractor_n,end_step=500_000):
    
    t_steps = 0
    max_rew = 0
    episode = 0
    #logs = pd.DataFrame(columns=['rew', 'max', 'ep'])
    while t_steps < end_step: # Training in number of steps
        episode += 1
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
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            # Adding s a r d to memory buffer #TODO:verify if good next_obs - obs
            agent.memory.remember(next_obs, action, reward, done)
            loss = agent.optimize_model()
            
            # obs = next_obs TODO: verify if good
            # update target network        
            if t_steps % agent.target_update_freq == 0:
                agent.update_target()
                print(f"Update Target Net : Episode: {episode + 1}, Epsilon: {agent.epsilon}")
                print(f'Loss at ts {t_steps} : {loss}')

            # save network weights
            # if save_model==True and t_steps % 100_000 == 0:
            #     model_save_path = f'dqn_breakout_regression_layer_{t_steps}.pth'
            #     torch.save(agent.regression_layer.state_dict(), model_save_path)
        
        if max_rew < total_reward:
            max_rew = total_reward
        print(f'Episode {episode}, reward = {total_reward}, max {max_rew}')
    
        
    model_save_path = f'ff_dqn_lastlayer_{feature_extractor_n}.pth'
    torch.save(agent.regression_layer.state_dict(), model_save_path)

    

#%%
if __name__ == '__main__':
    
    env = gym.make('BreakoutNoFrameskip-v4')
    env = BreakoutWrapper(env)

    # Initiate feature extractor with random agent
    print('Initalization, first network training')
    feature_extractor, features_size = init_feature_extractor(env,
                                                              batch_size=32, 
                                                              num_epochs=5, 
                                                              max_mem_size=100_000)
    #%%
    num_full_training = 10
    for training_n in range(num_full_training):
        # Create the agent (with the fixed feature extractor)
        ff_agent = FFAgent(feature_extractor, features_size,
                           memory_max_size=100_000)
        # Train the last layer
        print('Start training DQN')
        train_dqn_lastlayer(ff_agent, env,
                            training_n,
                            end_step=300_000)
        print('End of full training nb :', training_n)
        
        print('Start training feature extractor')
        feature_extractor, feature_size = train_feature_extractor(feature_extractor, 
                                                                  ff_agent.memory, # memory state to train feature extractor
                                                                  batch_size=32, 
                                                                  num_epochs=5)
    
