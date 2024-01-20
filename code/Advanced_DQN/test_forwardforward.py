#%%
import gym
import pandas as pd
import torch
import random

import matplotlib.pyplot as plt

from env_wrapper import BreakoutWrapper
from train_forwardforward import FFAgent, init_feature_extractor
from FF_CNN import FFConvNet


def test(agent, env, nb_episode=10):
    t_steps=0
    for episode in range(nb_episode): # Training in number of steps
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
            with torch.no_grad():
                state = state.unsqueeze(0) 
                state_features = agent.feature_extractor.respresentation_vects(state)
                action = agent.act(state_features)# Agent selects action
                
            next_obs, reward, done, _, reward_unclipped = env.step(action)
            env.render()
            total_reward += reward_unclipped
            # Adding s a r d to memory buffer #TODO:verify if good next_obs - obs
            agent.memory.remember(next_obs, action, reward, done)
    env.reset()
    return total_reward

def test_randomagent(env, nb_episode=10):
    t_steps = 0
    for episode in range(nb_episode):  # You can adjust the number of episodes for testing
        print('Start testing episode:', episode)
        obs, _ = env.reset()
        agent.memory.remember(obs, 0, 0, True)
        total_reward = 0
        done = False
        step = 0
        while not done:
            step += 1
            t_steps += 1
            action = random.randint(0, agent.action_size - 1)
            next_state, reward, done , _ , reward_unclipped = env.step(action)
            env.render()
            
            
        print(f'Total Reward for testing episode {episode}: {total_reward}')
    
    env.close()

#%%
if __name__ == '__main__':

    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    #env2 = gym.make('BreakoutNoFrameskip-v4')
    env = BreakoutWrapper(env)
    # load feature extractor
    feature_size = 103_808 # TODO: stop hardcoding this value
    feature_extractor_weights = 'ffconvnet_model_1.pth'
    
    feature_extractor = FFConvNet()
    feature_extractor.load_state_dict(torch.load(feature_extractor_weights))
    # Init agent
    agent = FFAgent(feature_extractor, feature_size,
                    memory_max_size=100_000,
                    epsilon=0)
    
    regression_layer_weights = 'ff_dqn_lastlayer_1.pth'
    agent.regression_layer.load_state_dict(torch.load(regression_layer_weights))
   
    total_reward = test_randomagent(env, nb_episode=15)
    env.close()
    