import gym
import time 

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import QNetwork
from env_wrapper import BreakoutWrapper
from replay_memory import ReplayBuffer

from train_backpropagation import DQNAgent, test



env2 = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
env2 = BreakoutWrapper(env2)
agent2 = DQNAgent()
pth_path = 'dqn_breakout_q_network_thebest.pth'
test(agent2, pth_path, env2, save_video=False, render=True)
