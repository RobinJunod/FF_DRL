#%%
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
 
models_dir = "DQN"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
   
env = make_atari_env(
    "BreakoutNoFrameskip-v4",
    n_envs=1,
    seed=42,
)
env = VecFrameStack(env, n_stack=4)
env.reset()
 
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=32,
    learning_starts=100000,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    optimize_memory_usage=False,
    verbose=0,
    tensorboard_log=logdir,
)


TIMESTEPS = 10000
iters = 0
for i in range(1000):
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")



#%% Adaptation of the FF in DQN
# create offpolicy algorithm class
# create a replay buffer
# Fill the buffer with random experiences
# Train the feature extractor using forward-forward and the replay buffer$
# LOOOP 
#    # Know create a DQN object with the FF feature extractor
#    # Make it a single layer and train it with DQN



import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import TrainFreq
from stable_baselines3.common.callbacks import BaseCallback


models_dir = "DQN"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
   
env = make_atari_env(
    "BreakoutNoFrameskip-v4",
    n_envs=1,
    seed=42,
)
env = VecFrameStack(env, n_stack=4)
env.reset()

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=32,
    learning_starts=100000,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    optimize_memory_usage=False,
    verbose=0, 
    tensorboard_log=logdir,
)

replay_buffer = ReplayBuffer(buffer_size=10_000,
                             observation_space=env.observation_space,
                             action_space=env.action_space)

train_freq = TrainFreq(frequency=10, unit="step")

base_callback = BaseCallback()

model.collect_rollouts(env=env, 
                       callback=base_callback, 
                       train_freq=train_freq, 
                       replay_buffer=replay_buffer)
#model0 = OffPolicyAlgorithm("CnnPolicy", env)
#model.collect_rollouts()
