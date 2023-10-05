# TODO : make an automatic experiments system to tune hyperparameters
# TODO : create a negative data generator 
#           1/ by playing random and switching the dimensionality between steps
#           2/ by playing randomfitting a gaussian model and generating points (neg points)
# TODO : improving the plotting, to also have the variance plot
#%%
import argparse

import torch
import random
import gym
from gym import wrappers

from FF_network import Net
from plotting_tools import moving_average, linear_graph

# TODO : create a metric to have the epsilon greedy rate

def FakeDataGenerator():
    # TODO: create 2 types of negative data generators
    pass

def DRL_train_network(
                    env, 
                    ff_net,
                    memory_capacity = 10000,
                    num_episodes=400,
                    num_epochs = 100,
                    epsilon_start = 1.0,
                    epsilon_end = 0.01,
                    epsilon_decay = 0.997,
                    theta_start = 5,
                    theta_end = 25,
                    theta_decay = 1.05,
                    ):
            
    # Initialize epsilon for epsilon-greedy exploration
    epsilon = epsilon_start
    # Initialize theta for negative data creation
    theta = theta_start

    # Initialize replay memory for good moves
    episode_memory = []
    # Initialize replay memory for good moves
    replay_memory_positive_list = []
    # Initialize replay memory for bad moves
    replay_memory_negative_list = []
    
    # Reward evolution 
    reward_evolution = [] 
    # Episode length evolution
    ep_length_evolution = []


    #% Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        episode_length = 0

        # This loop plays an episode until the agent dies
        while not done and episode_length < 2000:
            episode_length += 1
            # Epsilon-greedy exploration
            if random.random() < epsilon:
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
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        theta = min(theta_end, theta * theta_decay)
        
        # sort the replay memory such that the good runs stays in it (for better estimation of pos and neg data)
        replay_memory_positive_list.append(episode_memory[:-int(theta)])
        replay_memory_positive_list = sorted(replay_memory_positive_list, key=len, reverse=True)
        replay_memory_positive = [item for sublist in replay_memory_positive_list for item in sublist]
        if len(replay_memory_positive_list) > 100:
            replay_memory_positive_list.pop()
    
        replay_memory_negative_list.append(episode_memory[-int(theta):])
        replay_memory_negative = [item for sublist in replay_memory_negative_list for item in sublist]
        if len(replay_memory_negative_list) > 200:
            replay_memory_negative_list.pop()
            
        # Clear the replay memory of 1 run
        if len(replay_memory_positive) > memory_capacity:
            # Simple memory inside an array
            for _ in range(len(replay_memory_positive)-memory_capacity):
                # pop the last item of the list which should be the smallest run
                replay_memory_positive.pop()
            
        if len(replay_memory_negative) > memory_capacity:
            # Simple memory inside an array
            for _ in range(len(replay_memory_negative)-memory_capacity):
                # pop the last item of the list which should be the smallest run
                replay_memory_negative.pop()
                
        episode_memory.clear() # clear the memory of the past episode
        

        # Selecting k random sample in pos/neg memory data
        # x_pos has to be in a tensor
        neg_selection = random.choices(replay_memory_negative, k=256)
        # x_pos and x_neg must be tensor
        x_neg = torch.stack(neg_selection)
        if replay_memory_positive: # early stage when no pos data
            pos_selection = random.choices(replay_memory_positive, k=256)
            # x_pos and x_neg must be tensor
            x_pos = torch.stack(pos_selection)
        else:
            # Handle the case when replay_memory_positive is empty
            pos_selection = []        
        
        
        # Train the net if their is enough data
        if pos_selection and neg_selection:
            print('--------------start the training-----------------')
            print('replay pos mem list :',[len(inner_list) for inner_list in replay_memory_positive_list])
            #print('replay neg mem list :',[len(inner_list) for inner_list in replay_memory_negative_list])
            ff_net.train(x_pos,x_neg, num_epochs=num_epochs)
       
        
        # Log graph and plot outputs
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        print(f'length of repNeglist: {len(replay_memory_negative_list)} length of repNeg: {len(replay_memory_negative)} length of reppos: {len(replay_memory_positive)}')
        #print(f'length of the first repposlist list {len(replay_memory_positive_list[0])}')
        
        # Reward evolution
        reward_evolution.append((episode, total_reward))
        # Episode length evolution
        ep_length_evolution.append((episode, episode_length))


 
    logs = (reward_evolution, ep_length_evolution)
    
    # Close the environment
    env.close()
    
    return ff_net, logs

#%%
if __name__ == '__main__':
    # __________PARSERS__________
    parser = argparse.ArgumentParser(description="A simple command-line parser.")
    # Add command-line arguments
    parser.add_argument("--num_episodes", "-i", help="Input file path", required=False)
    parser.add_argument("--num_epochs", "-o", help="Output file path", required=False)
    parser.add_argument("--epsilon_start", "-v", help="Epsilon greedy start")
    parser.add_argument("--epsilon_decay", "-v", help="Epsilon greedy decay value after each episode")
    parser.add_argument("--epsilon_end", "-v", help="Epsilon greedy end")
    
    parser.add_argument("--theta_start", "-v", help="theta, the death horizon start")
    parser.add_argument("--theta_decay", "-v", help="theta, the death horizon decay value after each episode")
    parser.add_argument("--theta_end", "-v", help="theta, the death horizon end")
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Access the parsed arguments
    input_file = args.input_file
    output_file = args.output_file
    verbose = args.verbose
    
    # Forward Forward algo 
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0] + 1
    print(f'input size : {input_size}')
    # Create the forward forward network
    ff_net =  Net([input_size, 50, 20, 20])
    ff_net_trained, logs = DRL_train_network(env, ff_net, num_episodes=1000)
    
    # plot the logs
    reward_evolution, ep_length_evolution = logs

    # plot the reward evolution
    linear_graph(reward_evolution, 'Episode', 'Reward', 'Evolution of the Reward')
    moving_average(reward_evolution,x_axis_name='Episode',y_axsis_name='Reward', title='Adaptative negative data',window_size=100)
        