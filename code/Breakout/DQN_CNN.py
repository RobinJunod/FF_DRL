# %%
import gym
import ale_py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

from collections import deque
from  model import QNetwork

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Hyperparameters
        self.memory = deque(maxlen=50000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.000001 # Linear decay
        self.lr = 0.00025
        # Deep Neural Network
        self.q_network = QNetwork(action_size)
        self.target_q_network = QNetwork(action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        # Images Preprocessing
        self.transforms = T.Compose([T.ToPILImage(), T.Resize((84, 84)), T.Grayscale(), T.ToTensor()])
        self.last_frame = 0

    def preprocess_frame(self, frame):
        """Preprocessing of the frame used to remove flickering (nature paper)
        Args:
            frame (rgb  img): frame given from the env
        Returns:
            _type_: _description_
        """
        frame = np.maximum(frame, self.last_frame)
        self.last_frame = frame
        prepoc_frame = self.transforms(frame)
        return prepoc_frame
    
    def stack_states(self, states):
        # Stack the consecutive frames as a single 4-channel
        return torch.cat(states, dim=0)
    
    def remember(self, state, action, reward, next_state, done):
        # Append (s,a,r,s') to replay memory
        self.memory.append((state, action, reward, next_state, done))

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
    def act(self, stacked_state):
        """Select an action based on the Epsioln-greedy method
        Args:
            stacked_state (tensor 4x84x84): The input of the network (4 img preproc)
        Returns:
            action: The action that must be played
        """
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            stacked_state_ = stacked_state.unsqueeze(0)
            q_values = self.q_network(stacked_state_)
            action = torch.argmax(q_values).item()
            return action
        
    def optimize_model(self):
        # Here states and next_states are 4 succ img in grey scale
        if len(self.memory) < self.batch_size:
            return
        # Get samples from replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        dones = torch.tensor(dones, dtype=torch.int)
        # Get all the Q values at each states 
        q_values = self.q_network(states)
        # The next Q values are determined by the more stable network (off policy)
        next_q_values = self.target_q_network(next_states).max(1).values
        # The target q value is the computed as the sum of the reward and the futur best Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Select the Q values of the actions taken
        q_values = q_values.gather(1, actions.view(-1, 1))
        # Compute the loss from the q values differences (the prediction and the target) (2-arrays of length 32 (batch-size))
        loss = F.smooth_l1_loss(q_values, target_q_values.view(-1, 1))
        # Optimization using basic pytorch code
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Decrease the epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        

def train(agent, env, nb_epsiode=10):
    
    TARGET_UPDATE_FREQ = 10000
    t_steps = 0
    
    for episode in range(nb_epsiode):
        print('Start episode :', episode)
        frame, infos = env.reset()
        state = agent.preprocess_frame(frame)
        states = [state] * 4  # Initialize the stack with the same frame
        total_reward = 0
        done = False
        terminated = False
        step = 0

        while not done and not terminated:#  and step < 300:
            step += 1
            t_steps += 1
            #print('state :', state.shape)
            stacked_state = agent.stack_states(states)  # Stack the frames            
            action = agent.act(stacked_state) # Agent selects action
            
            new_frame, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            next_state = agent.preprocess_frame(new_frame)
            states.pop(0)  # Remove the oldest frame
            states.append(next_state)  # Add the newest frame
            stacked_next_state = agent.stack_states(states) # Create stacked next frame
            
            # stacked_state/next_state : [4,84,84] , action/reward : int , done : bool
            agent.remember(stacked_state, action, reward, stacked_next_state, done)
            # print(f'stacked state{stacked_state.shape}, action{action}, reward{reward}, next {stacked_next_state.shape}, done{done}')
            
            agent.optimize_model()
            
            if t_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_q_network()
                print(f"Update Target Net : Episode: {episode + 1}, Epsilon: {agent.epsilon}")
            
            
        print(f'Total Reward over the episode:{total_reward} , Current epsilon : {agent.epsilon}')

    env.close()
    torch.save(agent.q_network.state_dict(), 'dqn_breakout_q_network.pth')
    
    
    
if __name__ == '__main__':
    # Initialize the Breakout environment
    env = gym.make('Breakout-v0')
    state_size = (1, 84, 4 * 84)  # Four consecutive frames as input
    action_size = env.action_space.n

    # Initialize the DQN agent
    agent = DQNAgent(state_size, action_size)

    # Train DQL
    train(agent,env)
    

