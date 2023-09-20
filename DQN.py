#%%
import gym
from gym import wrappers

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# plotting function
def plot_tuples(tuples_list, x_label, y_label, title):
    # Extract x and y values from the list of tuples
    x_values, y_values = zip(*tuples_list)
    # Create a line plot
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    # Add labels and a title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # Show the plot
    plt.show()


    

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.99
memory_capacity = 10000
batch_size = 32
target_update_frequency = 20

# Initialize the environment
env = gym.make("CartPole-v1")
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Initialize the Q-networks
q_network = QNetwork(input_size, output_size)
target_network = QNetwork(input_size, output_size)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Initialize replay memory
replay_memory = []

# Initialize epsilon for epsilon-greedy exploration
epsilon = epsilon_start

# Reward evolution 
reward_evolution = [] 
# Episode length evolution
ep_length_evolution = []
# Evolution of the loss
loss_evolution = []

#%% Training loop
num_episodes = 300
for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0
    episode_length = 0
    loss_mean = 0
    while not done and episode_length < 2000:
        episode_length += 1
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            with torch.no_grad():
                q_values = q_network(state)
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


#%%
def test_policy(env, q_network):
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    t = 0
    while not done and t < 500:
        t += 1
        
        # take the policy of the state
        q_values = q_network(state)
        # take the best action given the q_values
        action = torch.argmax(q_values).item()
        #Take the selected action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.tensor(next_state, dtype=torch.float32)
        #if done:
        #   break
        state = next_state
        env.render()
        
    env.close()
    return t


#%%
if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    test_policy(env, q_network)
    #env = gym.make('CartPole-v1', render_mode='rgb_array')
    ## play a final set of episodes
    ##env = wrappers.Monitor(env, 'my_awesome_dir', force='True')
    ##env = wrappers.RecordVideo(env, 'my_awesome_dir', video_callable=lambda episode_id: True, codec='x264')
    #env = wrappers.RecordVideo(env, 'my_awesome_dir')
    #print("***Final run with final weights***:", play_policy(env, q_network))
    


# %%
