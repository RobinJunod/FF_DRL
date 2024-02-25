
# The feature extractor weights should then be fixed, and a trainable linear layer is added on top of it
# The new model (extractor+linear layer) is trained as a standard DQN
# Experience (i.e. positive data) is generated with the DQN and used to generate new negative data
# Then the feature extractor is trained with these data
# Repeat 1-4

#%%
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import gym
from gym import wrappers
from gym.utils.save_video import save_video

from FF_regression import Feature_extractor
from Dataset import fake_data_shuffle
from plotting_tools import  moving_average


def play_random(env, num_episodes = 3000):
    """Play with random actions
    Args:
        env (_type_): the gym environment
    Returns:
        (states, rewards): a matrix of all states it has been through,
                           a list of the rewards for random episodes
    """
    memory_capacity = 10_000
    episode_memory = []
    rnd_action_rewards = []
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        # Evaluation Variables
        episode_length = 0
        total_reward = 0
        # This loop plays an episode until the agent dies
        while not done and episode_length < 2000:
            episode_length += 1
            action = env.action_space.sample()  # Random action

            # Take the selected action (New API)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # New API, the done flag can be detected wether the episode failed or succed
            done = terminated or truncated
            # put in a torch tensor
            next_state = torch.tensor(next_state, dtype=torch.float32)
            # Just store the state action pair
            episode_memory.append(state)
            if len(episode_memory) > memory_capacity:
                episode_memory.pop()
            total_reward += reward
            state = next_state
        rnd_action_rewards.append(total_reward)
        
    real_states = torch.stack(episode_memory)
    # Close the environment
    env.close()
    
    return real_states, rnd_action_rewards


def DQL(env, feature_extractor, feature_size, num_episodes=300, memory_capacity = 30000):
    # Hyperparameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.97
    
    batch_size = 64
    target_update_frequency = 20

    # Regression last layer (not OLS because RL)
    regression_layer = nn.Linear(feature_size, env.action_space.n)
    # Target init    
    target_regression_layer = nn.Linear(feature_size, env.action_space.n)
    target_regression_layer.load_state_dict(regression_layer.state_dict())
    target_regression_layer.eval()
    
    # Create to try without feature extractor
    # regression_layer0 =  nn.Linear(env.observation_space.sample().size, env.action_space.n)
    # target_regression_layer0 = nn.Linear(env.observation_space.sample().size, env.action_space.n)
    # target_regression_layer0.load_state_dict(regression_layer0.state_dict())
    # target_regression_layer0.eval()
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(regression_layer.parameters(), lr=0.0005) # TODO : change to feature model
        

    # Initialize epsilon for epsilon-greedy exploration
    epsilon = epsilon_start
    # Initialize replay memory
    replay_memory = []
    # Reward evolution 
    reward_evolution = []
    loss_evolution = []

    #% Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        episode_length = 0
        loss_mean_ep = 0
        while not done and episode_length < 2000:
            episode_length += 1
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():
                    state_features = feature_extractor.inference(state)
                    q_values = regression_layer(state_features)
                    #q_values = regression_layer(state) # TODO : remove after
                    action = torch.argmax(q_values).item()
            # Take the selected action (New API)
            next_state, reward, terminated, truncated, info = env.step(action)
            # put in a torch tensor
            next_state = torch.tensor(next_state, dtype=torch.float32)
            # New API, the done flag can be detected wether the episode failed or succed
            done = terminated or truncated


            # Store the transition in replay memory
            replay_memory.append((state, action, reward, next_state, done))
            if len(replay_memory) > memory_capacity:
                # Simple memory inside an array
                replay_memory.pop(0)

            # Sample a random batch from replay memory and perform Q-learning update
            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                # stack the informations buffer size tensor (for a better traing in backward prop)
                states = torch.stack(states)
                next_states = torch.stack(next_states)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                dones = torch.tensor(dones, dtype=torch.float32)
                # use the network to estimate the next q values given the buffer of states
                # take a bunch of data to stabilize the learning
                q_values = regression_layer(feature_extractor.inference(states))
                #q_values = regression_layer(states) #TODO: remove after
                # The next q values are determined by the more stable network (off policy)
                next_q_values = target_regression_layer(feature_extractor.inference(next_states)).max(1).values
                #next_q_values = target_regression_layer(next_states).max(1).values # TODO remove after
                # The target q value is the computed as the sum of the reward and the futur best Q values
                target_q_values = rewards + (1 - dones) * gamma * next_q_values
                # takes the values of the action choosen by the epsilon greedy !! (so we have a q_value of dim 2 projecting to dim 1 (only retaining the best))
                # It is a bit like taking the 'V-value' of the Q-value by selecting the best action
                q_values = q_values.gather(1, actions.view(-1, 1))
                # compute the loss from the q values differences (the prediction and the target) (2-arrays of length 32 (batch-size))
                loss = criterion(q_values, target_q_values.view(-1, 1))
                # Optimization using basic pytorch code
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Add the loss evolution
                loss_mean_ep += loss_mean_ep*(episode_length-1)/episode_length + loss.item()/episode_length
                
            total_reward += reward
            state = next_state
            
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        # Update the target network
        if episode % target_update_frequency == 0:
            target_regression_layer.load_state_dict(regression_layer.state_dict())#TODO change to feature
            target_regression_layer.eval()
            
        # Print part
        print(f"Episode {episode + 1}, Total Reward: {total_reward}") if episode%10 == 0 else None
        # Reward evolution
        reward_evolution.append((episode, total_reward))
        loss_evolution.append((episode, loss_mean_ep))
        
    # plot the reward evolution
    # moving_average(loss_evolution, x_axis_name='episode', y_axsis_name='loss', title='Moving average plot', window_size=20)
    moving_average(reward_evolution, x_axis_name='episode', y_axsis_name='Reward', title='Reward Evolution', window_size=50)

    # Thoses real states will be used to train the feature extractor
    states, actions, rewards, next_states, dones = zip(*replay_memory)
    real_states = torch.stack(states)
    return real_states, regression_layer, reward_evolution


def test_policy(env, feature_extractor, regression_layer, save_vid=True):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    t = 0
    while not done and t < 500:
        t += 1
        
        with torch.no_grad():
            state_features = feature_extractor.inference(state)
            q_values = regression_layer(state_features)
            action = torch.argmax(q_values).item()
        
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


    save_model=True
    save_logs=False
    
    # Forward Forward algo 
    env = gym.make("CartPole-v1")
    # Initialize Feature extractor
    input_size = env.observation_space.shape[0]
    # Initalize the postiive adn negative data
    positive_data, _ = play_random(env)
    positive_data, negative_data = fake_data_shuffle(positive_data)    

    # To save the model
    Best_FE_score = 0
    Best_FE = None
    Best_LL = None
    
    # Train feature extractor
    feature_extractor = Feature_extractor([input_size, 20, 10, 10])  
    for ff_train in range(8):
        print ('New feature extractor training n° ', ff_train)
        feature_extractor.train_L(positive_data, negative_data, num_epochs=150)
        # Train last layer and get new states
        # Initialize Q one layer net
        feature_size = len(feature_extractor.inference(positive_data)[0])
        #regression_layer = nn.Linear(feature_size, env.action_space.n)
        positive_data, regression_layer, reward_evolution = DQL(env, feature_extractor, feature_size, num_episodes=350)
        positive_data, negative_data = fake_data_shuffle(positive_data)

        # Create a Pandas DataFrame from the lists
        df1 = pd.DataFrame(reward_evolution, columns=['episode_R', 'Reward Evolution'])
        # Save the DataFrames to a CSV file
        if save_logs:
            df1.to_csv(f'../results/FF_Qlearning/logs/reward_config_{ff_train}_{len(positive_data)}.csv', index=False)
        # Save the best FE for the inference part
        sum_last_10_total_rewards = sum([total_reward for _, total_reward in reward_evolution[-10:]])
        if sum_last_10_total_rewards > Best_FE_score:
            Best_FE_score = sum_last_10_total_rewards
            Best_FE = copy(feature_extractor)
            Best_LL = copy(regression_layer)
    
    torch.save(Best_FE.state_dict(), f'DQN_FeatureExtractor_bestweights.pth')
    torch.save(Best_LL.state_dict(), f'DQN_RegressLayer_bestweights.pth')
    env.close()
    # Saving the experiment
    
    
    #%% Test the policy and render it
    
    env = gym.make('CartPole-v1', render_mode='human')    
    test_policy(env, Best_FE, Best_LL, save_vid=False)

    
    ##%% RANDOM STUFF TO 
    #input_size = env.observation_space.shape[0] 
    ## Create the forward forward network
    #ff_net =  feature_extractor([input_size, 10, 10])
    ## Generate real states
    #real_states, rnd_action_rewards = play_random(env)
    ## Create fake states
    #real_data, fake_data = fake_data_shuffle(real_states)
    
    ## TEST FAKE DATA GENERATOR
    #train_realData_l = 0.8 * len(real_data)
    #train_fakeData_l = 0.8 * len(fake_data)
    #
    #train_realData = real_data[:int(train_realData_l)]
    #train_fakeData = fake_data[:int(train_fakeData_l)]
    #
    #test_realData = real_data[int(train_realData_l):]
    #test_fakeData = fake_data[int(train_fakeData_l):]
    #
    ## Train the network on these real and fake states
    #ff_net.train(train_realData, train_fakeData, num_epochs=100)
    ## Dataset split
    #pos_data_result = ff_net.predict(test_realData)
    #print('pos data: ', pos_data_result)
    #print('pos data mean: ', pos_data_result.mean())
    #print('pos data std: ', pos_data_result.std())
    #neg_data_result = ff_net.predict(test_fakeData)
    #print('neg data: ', neg_data_result)
    #print('neg data mean: ', neg_data_result.mean())
    #print('neg data std: ', neg_data_result.std())
    