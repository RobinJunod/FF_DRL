# TODO : make an automatic experiments system to tune hyperparameters
# TODO : create a negative data generator 
#           1/ by playing random and switching the dimensionality between steps
#           2/ by playing randomfitting a gaussian model and generating points (neg points)
# TODO : improving the plotting, to also have the variance plot
#%%
import random
import numpy as np
from sklearn.mixture import GaussianMixture

import torch
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

def FDGenerator_GGM(real_states, max_components_to_try=15, n_samples=256):
    """Fit a GMM to the real state data in order to create from thoses points some fake data points

    Args:
        real_states (torch matrix): matrix containing the real state value obersved
        max_components_to_try (int, optional): The number of gmm max to try to fit the dataset. Defaults to 10.
        n_samples (int, optional): number of fake samples generated. Defaults to 256.

    Returns:
        torch matrix: fake samples
    """
    data = real_states.numpy()
    
    lowest_bic = np.inf
    best_n_components = 0
    for n_components in range(1, max_components_to_try + 1):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)
        bic = gmm.bic(data)
        
        if bic < lowest_bic:
            lowest_bic = bic
            best_n_components = n_components
            
    # Fit the best model to our dataset
    best_gmm = GaussianMixture(n_components=best_n_components)
    best_gmm.fit(data)
    
    # Create n fake samples
    new_samples = best_gmm.sample(n_samples)
    # Put it in a torch tensor
    fake_states = torch.from_numpy(new_samples)

    return real_states, fake_states


def play_random(env):
    memory_capacity = 10000
    num_episodes = 1000
    episode_memory = []
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        # Evaluation Variables
        episode_length = 0
        
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
        
            state = next_state
        
    
    real_states = torch.stack(episode_memory)
    # Close the environment
    env.close()
    return real_states




#%%
if __name__ == '__main__':

    # Forward Forward algo 
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0] 
    
    # Create the forward forward network
    ff_net =  Net([input_size, 50, 20, 20])
    
    # Generate real states
    real_states = play_random(env)
    
    # Create fake states
    real_data, fake_data = Fake_data_shuffle(real_states)
    #%%
    # Train the network on these real and fake states
    ff_net.train(real_data, fake_data, num_epochs=100)



