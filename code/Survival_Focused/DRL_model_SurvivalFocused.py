# TODO : improving the plotting, to also have the variance plot
#%%
import argparse
import random
import pandas as pd
import numpy as np
import torch
import gym
from gym import wrappers
from gym.utils.save_video import save_video

from FF_network import Net
#from ..plotting_tools import moving_average, linear_graph


def DRL_train_network(env, ff_net, cell=False, **kwargs):
    if cell:
        memory_capacity =10_000
        num_episodes=3000
        num_epochs=50
        epsilon_start=1
        epsilon_decay=0.99
        epsilon_end=0.1
        theta_start=5
        theta_decay=1.05
        theta_end=25
    else:
        # Get the hyperparameters
        memory_capacity = kwargs.get("memory_capacity")
        num_episodes = kwargs.get("num_episodes")
        num_epochs = kwargs.get("num_epochs")
        epsilon_start = kwargs.get("epsilon_start")
        epsilon_end = kwargs.get("epsilon_end")
        epsilon_decay = kwargs.get("epsilon_decay")
        theta_start = kwargs.get("theta_start")
        theta_end = kwargs.get("theta_end")
        theta_decay = kwargs.get("theta_decay")

    
    # Initialize epsilon for epsilon-greedy exploration
    epsilon = epsilon_start
    # Initialize theta for negative data creation
    theta = theta_start
    # automatic theta and epsilon decay 
    theta_decay = np.exp(1/(0.6*num_episodes)*np.log(theta_end/theta_start))
    epsilon_decay = np.exp(1/(0.6*num_episodes)*np.log(epsilon_end/epsilon_start))
    
    print('theta decay : ', theta_decay, ' ---- epsilon decay : ', epsilon_decay)
    # Initialize replay memory for good moves
    episode_memory = []
    # Initialize replay memory for good moves
    replay_memory_positive_list = []
    # Initialize replay memory for bad moves
    replay_memory_negative_list = []
    # Reward evolution 
    reward_evolution = [] 
    # Episode length evolution
    exploration_rate_evolution = []


    #% Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        # Evaluation Variables
        total_reward = 0
        exploration_count = 0
        episode_length = 0
        
        # This loop plays an episode until the agent dies
        while not done and episode_length < 2000:
            episode_length += 1
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                exploration_count += 1
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():
                    # create intput state-action with default action being 0
                    input_1  = torch.cat((state, torch.tensor([1])), dim=0)
                    action_1 = ff_net.predict(input_1).item()
                    input_0  = torch.cat((state, torch.tensor([0])), dim=0)
                    action_0 = ff_net.predict(input_0).item()
                    action = 1 if action_1 > action_0 else 0
            # Take the selected action (New API)
            next_state, reward, terminated, truncated, info = env.step(action)
            # New API, the done flag can be detected wether the episode failed or succed
            done = terminated or truncated
            # put in a torch tensor
            next_state = torch.tensor(next_state, dtype=torch.float32)
            # Store the transition in replay memory
            # replay_memory.append((state, action, reward, next_state, done))
            # Just store the state action pair
            episode_memory.append(torch.cat((state, torch.tensor([action])), dim=0))
            total_reward += reward
            state = next_state


        # Epsilon and Theta decay
        epsilon = min(epsilon_end, epsilon * epsilon_decay) if epsilon_decay > 1 else max(epsilon_end, epsilon * epsilon_decay) 
        theta = min(theta_end, theta * theta_decay) if theta_decay > 1 else max(theta_end, theta * theta_decay)
        
        # Sort the replay memory such that the good runs stays in it (for better estimation of pos and neg data)
        replay_memory_positive_list.append(episode_memory[:-int(theta)])
        replay_memory_positive_list = sorted(replay_memory_positive_list, key=len, reverse=True)
        replay_memory_positive = [item for sublist in replay_memory_positive_list for item in sublist]
        if len(replay_memory_positive) > memory_capacity:
            replay_memory_positive_list.pop()
    
        replay_memory_negative_list.append(episode_memory[-int(theta):])
        replay_memory_negative = [item for sublist in replay_memory_negative_list for item in sublist]
        if len(replay_memory_negative) > memory_capacity:
            replay_memory_negative_list.pop()
        
        
        episode_memory.clear() # clear the memory of the past episode
        
        if replay_memory_positive and replay_memory_negative:
            # Selecting k random sample in neg memory data
            neg_selection = random.choices(replay_memory_negative, k=256)
            # x_pos and x_neg must be tensor
            x_neg = torch.stack(neg_selection)
            # Selecting k random sample in pos memory data
            pos_selection = random.choices(replay_memory_positive, k=256)
            # x_pos and x_neg must be tensor
            x_pos = torch.stack(pos_selection)

        
        # Train the net if their is enough data
        if pos_selection and neg_selection:
            #ff_train_type = 'LAYERS'
            ff_train_type = 'LAYERS'
            print(f'--------------start the training {ff_train_type}-----------------')
            # Select the type of training (details for experimentation, bio plaisible vs more logical one)
            if ff_train_type == 'LAYERS':

                ff_net.train_L(x_pos,x_neg, num_epochs=num_epochs) # training layers by layers
            elif ff_train_type == 'BATCHES':
                ff_net.train_B(x_pos,x_neg, num_epochs=num_epochs) # training batches, passing data trough all layers first
            
       
        
        # Log graph and plot outputs
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        # print(f'length of repNeglist: {len(replay_memory_negative_list)} length of repNeg: {len(replay_memory_negative)} length of reppos: {len(replay_memory_positive)}')
        
        # Reward evolution
        reward_evolution.append((episode, total_reward))
        # Epsilon greedy rate
        exploration_rate_evolution.append((episode, exploration_count/episode_length))

    
    logs = (reward_evolution, exploration_rate_evolution)
    
    # Close the environment
    env.close()
    
    return ff_net, logs


def test_policy(env, ff_net_trained, save_vid=True):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    t = 0
    while not done and t < 500:
        t += 1
        with torch.no_grad():
            input_1  = torch.cat((state, torch.tensor([1])), dim=0)
            action_1 = ff_net_trained.predict(input_1).item()
            input_0  = torch.cat((state, torch.tensor([0])), dim=0)
            action_0 = ff_net_trained.predict(input_0).item()
            action = 1 if action_1 > action_0 else 0
        
        #Take the selected action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.tensor(next_state, dtype=torch.float32)
        if done:
            if save_vid:
                save_video(
                    env.render(),
                    "../results/videos",
                    fps=env.metadata["render_fps"],
                    step_starting_index=0,
                    episode_index=0)
            break
        state = next_state
        #env.render()
    env.close()
    return t
#%%
if __name__ == '__main__':
    # __________PARSERS__________
    parser = argparse.ArgumentParser(description="A simple command-line parser.")
    # Add command-line arguments
    parser.add_argument("--memory_capacity", type=int, default=10000, help="memory_capacity in the pos and negative list")
    parser.add_argument("--num_episodes", type=int, default=200, help="Input file path")
    parser.add_argument("--num_epochs", type=int, default=50, help="Output file path")
    parser.add_argument("--epsilon_start", type=int, default=1, help="Epsilon greedy start")
    #parser.add_argument("--epsilon_decay", type=float, default=0.1**(1/1_500), help="Epsilon greedy decay value after each episode")
    parser.add_argument("--epsilon_end", type=int, default=0.1, help="Epsilon greedy end")
    parser.add_argument("--theta_start", type=int, default=5, help="theta, the death horizon start")
    #parser.add_argument("--theta_decay", type=float, default=1.05, help="theta, the death horizon decay value after each episode")
    parser.add_argument("--theta_end", type=int, default=25, help="theta, the death horizon end")
    parser.add_argument("--train_e", type=bool, default=False, help="theta, the death horizon end")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Create a dictionary of arguments and their values
    arguments = {
        "memory_capacity": args.memory_capacity,
        "num_episodes": args.num_episodes,
        "num_epochs": args.num_epochs,
        "epsilon_start": args.epsilon_start,
        "epsilon_decay": 0.1**(1/1_500),
        "epsilon_end": args.epsilon_end,
        "theta_start": args.theta_start,
        "theta_decay": (args.theta_end/args.theta_start)**(1/1_500),
        "theta_end": args.theta_end,
        "train_e": args.train_e,
    }
    #%% To use for cell running
    #arguments = {
    #    "memory_capacity":10_000,
    #    "num_episodes":3_000,
    #    "num_epochs":100,
    #    "epsilon_start":1,
    #    "epsilon_decay":0.1**(1/1_500),
    #    "epsilon_end":0.1,
    #    "theta_start":25,
    #    "theta_decay":(5/25)**(1/1_500),
    #    "theta_end":5,
    #}
    
    # Forward Forward algo 
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0] + 1
    
    # Create the forward forward network
    ff_net =  Net([input_size, 50, 20, 20])
    ff_net_trained, logs = DRL_train_network(env, ff_net, **arguments)
    

    
    #%% Plot the logs
    reward_evolution, exploration_rate_evolution = logs
    
    # Saving the experiment
    csv_file = f'config_{args.memory_capacity}_{args.num_episodes}_{args.theta_start}_{args.theta_end}.csv'
    #%% Create a Pandas DataFrame from the lists
    df1 = pd.DataFrame(reward_evolution, columns=['episode_R', 'Reward Evolution'])
    df2 = pd.DataFrame(exploration_rate_evolution, columns=['episode_E', 'Exploration Evolution'])
    # Save the DataFrames to a CSV file
    df1.to_csv('../../results/SF_experiments/reward_B_'+csv_file, index=False)
    df2.to_csv('../../results/SF_experiments/explo_B_'+csv_file, index=False)
    
    print(f'Data saved to {csv_file}')

    #%% Play
    #env_test = gym.make('CartPole-v1', render_mode='rgb_array_list')  
    #test_policy(env_test, ff_net_trained, save_vid=True)
