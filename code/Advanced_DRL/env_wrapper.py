#%%
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torchvision.transforms as T

"""This wrappe r sohuld create an gym like env that outputs for the atari games breakout V0
the state in a simple fashion,
Remove the action no-op so that their is 3 possible actions only.

obs == observation given from atari env
preprocess(obs) == after img reshape and crop

"""

class BreakoutWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4):
        super(BreakoutWrapper, self).__init__(env)
        self.stack_frames = stack_frames


    def reset(self):
        """Each time it is called, play action Fire to start new episode
        Returns:
            state: return the custom state
        """
        self.env.reset()
        obs, _, _, _, info = self.env.step(1)  # Fire to start new episode
        self.frames = [self.preprocess(obs)] * self.stack_frames  # Initialize frames with the first observation
        
        return self._get_state(), info

    def step(self, action):
        """Wrapper of the step method
        Args:
            action (): can take discret values 1,2,3 (noop, RIGHT, LEFT)
        Returns:
            tuple: custom state, reward, term, trunc, info
        """
        if action == 1:
            action = 2
        elif action ==2:
            action = 3
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.frames.pop(0)
        self.frames.append(self.preprocess(obs))

        if info['lives'] < 5:
            print('CATASTROPHIQUE, CEST PERDU')
            terminated = True
            reward = -0.5 # Custom reward
            

        return self._get_state(), reward, terminated, truncated, info

    def show_current_state(self):
        current_state = self._get_state()
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            axes[i].imshow(current_state[i, :, :], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Frame {i + 1}')

    def preprocess(self, obs):
        preprocess = T.Compose([T.ToPILImage(), 
                                self._custom_crop,
                                T.Resize((84, 84)), 
                                T.Grayscale(), 
                                T.ToTensor(),
                                lambda x: x.squeeze() # Squeeze the extra dimension
                                ])
        return preprocess(obs)
    
    
    def _get_state(self):
        if self.stack_frames > 1:
            # Stack the frames along the last dimension
            return torch.stack(self.frames, dim=0)
        else:
            # Return the last frame
            return self.frames[-1]

    def _custom_crop(self, img):
        return T.functional.crop(img, top=25, left=10, height=img.size[1] - 25, width=img.size[0] - 20)
    


#%%
if __name__=='__main__':
    
    env = gym.make('Breakout-v0', render_mode='rgb_array')
    env = BreakoutWrapper(env, stack_frames=4)
    env.reset()
    
    # make 10 time same action
    for _ in range(10):
        env.step(0)
    env.show_current_state()
    

        
    