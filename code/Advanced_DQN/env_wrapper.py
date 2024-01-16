#%%
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
"""This wrappe r sohuld create an gym like env that outputs for the atari games breakout V0
the state in a simple fashion,
Remove the action no-op so that their is 3 possible actions only.

obs == observation given from atari env
preprocess(obs) == after img reshape and crop

"""

class BreakoutWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BreakoutWrapper, self).__init__(env)
        self.action_space.n = 3
        self.pp_obs = 0

    def reset(self):
        """Each time it is called, play action Fire to start new episode
        Returns:
            state: return the custom state
        """
        self.env.reset()
        # Play a rnd number of nothing before starting in order to create rnd init
        no_op_steps = np.random.randint(1, 31)  # Randomly choose the number of "do nothing" steps (1 to 30)
        for _ in range(no_op_steps):
            self.env.step(0) # do nothing
        obs, _, _, _, info = self.env.step(1)  # Fire to start new episode
        self.pp_obs = self.preprocess(obs)
        return self.pp_obs, info

    def step(self, action):
        """Wrapper of the step method
        Args:
            action (): can take discret values 0,1,2 (noop, RIGHT, LEFT)
        Returns:
            tuple: custom state, reward, term, trunc, info
        """
        # Select the good actions, wrapper vs gym
        if action == 1:
            action = 2 # change it to match the env
        elif action == 2:
            action = 3
        
        
        total_rewards = 0
        total_rewards_unclipped = 0
        rem_lives = 5 # check if lost a point
        done=False
        for i in range(4): # play 2 time the same action
            if i == 3 : action=0
            obs, reward, _, _, info = self.env.step(action)
            rem_lives = info['lives']
            total_rewards_unclipped += reward
            reward = np.clip(reward, -3, 3)
            total_rewards += reward

        if rem_lives < 5:
            done = True
            total_rewards = 0 
        
        self.pp_obs = self.preprocess(obs)
        return self.pp_obs, total_rewards, done, info, total_rewards_unclipped

            
    def show_current_obs(self):
        """Show the images of the current state
        """
        if env.render_mode == 'rgb_array':
            plt.imshow(self.pp_obs, cmap='gray')
            plt.show()
        else: print('Render mode must be rgb_array to use show_current_obs in wrapper')


    def preprocess(self, image):
        """Reduce dim of the observation space, for memory efficiency
        """
        # Crop the image (adjust the values as needed)
        top, left = 30, 10
        height, width = image.shape[0] - 30, image.shape[1] - 20
        image = image[top:top + height, left:left + width]

        # Resize the image
        image = F.resize(Image.fromarray(image), (84, 84))
        # Pass through greyscale
        image = F.to_grayscale(image)
        
        # Convert to boolean (0 or 1)
        threshold_value = 0.01 # Set your threshold value here
        image = (np.array(image) / 255.0 > threshold_value).astype(bool)

        return image


def play_openai():
    import gym
    from gym.utils.play import play
    play(gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array"), keys_to_action={
                                                   "q": 3,
                                                   "w" : 2,
                                                  }, noop=0)
    

#%%
if __name__=='__main__':
    
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
    env = BreakoutWrapper(env)
    obs, info = env.reset()
    env.show_current_obs()
    
    for i in range(30):
        env.step(1)
        if i%4==0:
            env.show_current_obs()
    
    #%% play breakout with keyboard
    play_openai()
    #%%