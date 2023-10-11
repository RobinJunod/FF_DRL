# TODO : Then, to sum up:
# You can use the positive and negative data you created to train a feature extractor
# The feature extractor weights should then be fixed, and a trainable linear layer is added on top of it
# The new model (extractor+linear layer) is trained as a standard DQN
# Experience (i.e. positive data) is generated with the DQN and used to generate new negative data
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

from FF_network import Net

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


def play_random(env):
    """Play with random actions
    Args:
        env (_type_): the gym environment
    Returns:
        (states, rewards): a matrix of all states it has been through,
                           a list of the rewards for random episodes
    """
    memory_capacity = 10000
    num_episodes = 5000
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




#%%
if __name__ == '__main__':

    # Forward Forward algo 
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0] 
    
    # Create the forward forward network
    ff_net =  Net([input_size, 50, 20, 20])
    
    # Generate real states
    real_states, rnd_action_rewards = play_random(env)
    
    #%% Create fake states
    real_data, fake_data = Fake_data_shuffle(real_states)
    #%% Create fake states
    real_data, fake_data, log_like = FDGenerator_GGM(real_states)
    
    #%% Split train-test set
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
    