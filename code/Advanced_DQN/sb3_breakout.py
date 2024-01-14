#%%
import os

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
 
models_dir = "../DQN"
logdir = "../logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

print('create atari env sb3')
env = make_atari_env(
    "BreakoutNoFrameskip-v4",
    n_envs=1,
    seed=42,
)

print('create env stack frames')
env = VecFrameStack(env, n_stack=4)
env.reset()
 
print('create cnn model from sb3')
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    batch_size=32,
    learning_starts=100_000,
    target_update_interval=1_000,
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
# %%
