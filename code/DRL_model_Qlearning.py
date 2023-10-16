# TODO : Then, to sum up:
# You can use the positive and negative data you created to train a feature extractor
# The feature extractor weights should then be fixed, and a trainable linear layer is added on top of it
# The new model (extractor+linear layer) is trained as a standard DQN
# Experience (i.e. positive data) is generated with the DQN and used to generate new negative data
# Then the feature extractor is trained with these data
# Repeat 1-4

# Make the plot for 
#%%
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture

import gym
from gym import wrappers

from FF_network_regression import Feature_extractor, Regression_Layer

from plotting_tools import moving_average, linear_graph

def Fake_data_shuffle(real_data_tensor):
    """Schuffle the data randomely to create fake data.
    The data dimensio are kept , only the values are changing
    Args:
        real_data_tensor (2d torch tensor): the "xpos", a 2d pytorch tensor (multiples state action pairs) 
    Returns:
        2d torch tensor : the "xpos", a 2d pytorch tensor (multiples state action pairs) 
    """
    
    # Transpose the tensor
    real_data_tensor_T_list = real_data_tensor.t().tolist()
    # Shuffle the values in each inner list
    for inner_list in real_data_tensor_T_list:
        random.shuffle(inner_list)
    
    fake_data_tensor = torch.tensor(real_data_tensor_T_list).T

    
    # Create a mask to remove the data that are similar 
    mask = torch.all(fake_data_tensor == real_data_tensor, dim=1)
    
    # Use the mask to remove rows from both tensors
    real_data= real_data_tensor[~mask]
    fake_data= fake_data_tensor[~mask]
    
    return real_data, fake_data

def FDGenerator_GGM(real_states, max_components_to_try=10):
    """Fit a GMM to the real state data in order to create from thoses points some fake data points

    Args:
        real_states (torch matrix): matrix containing the real state value obersved
        max_components_to_try (int, optional): The number of gmm max to try to fit the dataset. Defaults to 10.
        n_samples (int, optional): number of fake samples generated. Defaults to 256.

    Returns:
        torch matrix: fake samples
    """
    n_samples = len(real_states)
    data = real_states.numpy()
    
    lowest_bic = np.inf
    best_n_components = 0
    # Fit GMMs for different component numbers and calculate AIC/BIC
    aic_scores = []
    bic_scores = []
    log_likelihoods = []
    for n_components in range(1, max_components_to_try + 1):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)
        bic = gmm.bic(data)
        # TODO : choose and elbow/bic etc and justify it
        aic_scores.append(gmm.aic(data))
        bic_scores.append(gmm.bic(data))
        # if bic < lowest_bic:
        #     lowest_bic = bic
        #     best_n_components = n_components
        log_likelihoods.append(gmm.score_samples(data).mean())
    
    plt.plot(range(1, max_components_to_try + 1), log_likelihoods, label='log_likelihoods')
    #plt.plot(range(1, max_components_to_try + 1), aic_scores, label='AIC')
    #plt.plot(range(1, max_components_to_try + 1), bic_scores, label='BIC')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.legend()
    plt.show()        
    # Fit the best model to our dataset
    print(' Number of kernels :', best_n_components)
    best_gmm = GaussianMixture(n_components=3)
    best_gmm.fit(data)
    
    # Create n fake samples
    new_samples = best_gmm.sample(n_samples)[0]
    
    # Put it in a torch tensor
    fake_states = torch.from_numpy(new_samples).float()

    return real_states, fake_states, log_likelihoods


def play_random(env, num_episodes = 3000):
    """Play with random actions
    Args:
        env (_type_): the gym environment
    Returns:
        (states, rewards): a matrix of all states it has been through,
                           a list of the rewards for random episodes
    """
    memory_capacity = 10000
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
            next_state, reward, terminated, truncated, info = env.step(action)
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


def DQL(env, ff_network):
    # Hyperparameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99
    memory_capacity = 10000
    target_update_frequency = 20
    
    # Initialize Feature extractor
    input_size = env.observation_space.shape[0]
    feature_extractor = Feature_extractor([input_size, 20, 10, 10, 10])
    # Initalize the postiive adn negative data
    positive_data, _ = play_random(env)
    negative_data = Fake_data_shuffle(positive_data)
    feature_extractor.train(positive_data, negative_data, num_epochs=100)
    
    # Initialize Q one layer net
    size_feature = len(feature_extractor.inference(positive_data)[0])
    regression_Layer= Regression_Layer(size_feature, env.action_space.shape[0])
    target_regression_Layer = Regression_Layer(size_feature, env.action_space.shape[0])
    target_regression_Layer.load_state_dict(regression_Layer.state_dict())
    target_regression_Layer.eval()
    

        

    # Initialize epsilon for epsilon-greedy exploration
    epsilon = epsilon_start
    # Initialize replay memory
    replay_memory = []
    # Reward evolution 
    reward_evolution = []


    #% Training loop
    num_episodes = 200
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        episode_length = 0
        while not done and episode_length < 2000:
            episode_length += 1
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():
                    q_values = regression_Layer(feature_extractor.inference(state))
                    action = torch.argmax(q_values).item()
            # Take the selected action (New API)
            next_state, reward, terminated, truncated, info = env.step(action)
            # New API, the done flag can be detected wether the episode failed or succed
            done = terminated or truncated

            # put in a torch tensor
            next_state = torch.tensor(next_state, dtype=torch.float32)

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
                q_values = q_network(states)
                # The next q values are determined by the more stable network (off policy)
                next_q_values = target_network(next_states).max(1).values
                # The target q value is the computed as the sum of the reward and the futur best Q values
                # for cartpole the reward is 1 at each step 
                target_q_values = rewards + (1 - dones) * gamma * next_q_values
                # takes the values of the action choosen by the epsilon greedy !! (so we have a q_value of dim 2 projecting to dim 1 (only retaining the best))
                # It is a bit like taking the 'V-value' of the Q-value by selecting the best action
                q_values = q_values.gather(1, actions.view(-1, 1))
                
                # compute the loss from the q values differences (the prediction and the target) (2-arrays of length 32 (batch-size))
                loss = criterion(q_values, target_q_values.view(-1, 1))
                # loss mean evolution (recursive mean)
                loss_mean = ((episode_length-1)*loss_mean + loss.item())/episode_length
                
                # Optimization using basic pytorch code
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            # Update the target network
            if episode % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
                target_network.eval()

            total_reward += reward
            state = next_state

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # Reward evolution
        reward_evolution.append((episode, total_reward))
        # Episode length evolution
        ep_length_evolution.append((episode, episode_length))
        # Loss evolution
        loss_evolution.append((episode, loss_mean))

    # plot the reward evolution
    plot_tuples(reward_evolution, 'Episode', 'Total Reward', 'Reward Evolution Over Episodes')
    # plot the reward evolution
    plot_tuples(ep_length_evolution, 'Episode', 'Episode Length', 'Evolution of the Episode Lengths')
    # plot the reward evolution
    plot_tuples(loss_evolution, 'Episode', 'Loss mean', 'Evolution of the Loss')


    # Close the environment
    env.close()
    pass

#%%
if __name__ == '__main__':

    # Forward Forward algo 
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0] 
    
    # Create the forward forward network
    ff_net =  Feature_extractor([input_size, 20, 10, 10, 10, 10])
    
    # Generate real states
    real_states, rnd_action_rewards = play_random(env)
    
    #%% Create fake states
    real_data, fake_data = Fake_data_shuffle(real_states)
    #%% Create fake states
    real_data, fake_data, log_like = FDGenerator_GGM(real_states)
    
    #%% TEST FAKE DATA GENERATOR
    train_realData_l = 0.8 * len(real_data)
    train_fakeData_l = 0.8 * len(fake_data)
    
    train_realData = real_data[:int(train_realData_l)]
    train_fakeData = fake_data[:int(train_fakeData_l)]
    
    test_realData = real_data[int(train_realData_l):]
    test_fakeData = fake_data[int(train_fakeData_l):]
    
    # Train the network on these real and fake states
    ff_net.train(train_realData, train_fakeData, num_epochs=100)
    
    # Dataset split
    #%%
    pos_data_result = ff_net.predict(test_realData)
    print('pos data: ', pos_data_result)
    print('pos data mean: ', pos_data_result.mean())
    print('pos data std: ', pos_data_result.std())
    
    
    neg_data_result = ff_net.predict(test_fakeData)
    print('neg data: ', neg_data_result)
    print('neg data mean: ', neg_data_result.mean())
    print('neg data std: ', neg_data_result.std())
    