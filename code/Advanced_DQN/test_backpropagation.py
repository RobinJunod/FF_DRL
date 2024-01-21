#%%
import gym
import pandas as pd
import torch
import random

from env_wrapper import BreakoutWrapper
from train_backpropagation import DQNAgent


def test(agent, pth_path, env, nb_epsiode=10, render=True):
    # Load the trained model
    agent.q_network.load_state_dict(torch.load(pth_path))
    agent.q_network.eval()
    t_steps = 0
    logs = pd.DataFrame(columns=['episode','rew'])
    for episode in range(nb_epsiode):  # You can adjust the number of episodes for testing
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
            next_state, reward, done , _, reward_unclipped = env.step(action)
            total_reward += reward_unclipped
            agent.memory.remember(next_state, reward, action, done)
            if render:
                env.render()
        
        new_row = { 'episode' : episode,
                    'rew': total_reward}
        logs.loc[len(logs)] = new_row
            
        print(f'Total Reward for testing episode {episode}: {total_reward}')
        
    logs.to_csv(f'logs/test_{nb_epsiode}ep.csv', index=False)
    env.close()

def test_randomagent(env, nb_epsiode=10, save_video=False, render=False):
    t_steps = 0
    logs = pd.DataFrame(columns=['episode','rew'])
    for episode in range(nb_epsiode):  # You can adjust the number of episodes for testing
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
            total_reward += reward_unclipped
            agent.memory.remember(next_state, reward, action, done)
            if render:
                env.render()
        
        new_row = { 'episode' : episode,
                    'rew': total_reward}
        logs.loc[len(logs)] = new_row
            
        print(f'Total Reward for testing episode {episode}: {total_reward}')
        
    logs.to_csv(f'logs/test_rndAgent_{nb_epsiode}ep.csv', index=False)
    env.close()

def plot_results():
    pass

#%%
if __name__ == '__main__':
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    #env2 = gym.make('BreakoutNoFrameskip-v4')
    env = BreakoutWrapper(env)
    agent = DQNAgent()
    pth_path = 'model_weights_BP/dqn_breakout_q_network_thebest.pth'
    test(agent, pth_path, env, nb_epsiode=10)
