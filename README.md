# Alternative to backpropagation for Reinforcement  Learning

Forward-Froward Algorithm from G. Hinton used for Reinforcement learning.
This is a semester project at the EPFL, Master (robotic/datascience).

Keywords : Breakout , Forward-Forward, Cartpole, Atari games, DQN ,Double DQN, replay memeory

Autors Robin Junod, supervised by Giulio Romanelli. 

<img src="results/videos/gif_breakout.gif" alt="Cartpole FF" width="150%">


## Installation and depedencies
packages needed: pytorch, sklearn, pandas, numpy, gymnasium, gym atari (New version), matplotlib
To have the exact same env : `conda env create -f environment.yml`, `conda activate my_project_env`

# Part 1 : survival-focused
Promissing results have been achieved with a really straightfowrad implementation on the simple cartpole env. Go in the folder code/Survival_Focused

## Run the training phase
Run the fowrad-forward network train on vanilla ds : `python FF_network.py`

Run training for Surv-Focused in bash: `./run_exp.sh`

Or Run the python file directly : `python DRL_model_SurvivalFocused.py`

### Command-line Arguments

- `--memory_capacity`: (Type: int, Default: 10000) Memory capacity in the positive and negative list.
- `--num_episodes`: (Type: int, Default: 200) Number of episodes.
- `--num_epochs`: (Type: int, Default: 50) Number of epochs.
- `--epsilon_start`: (Type: int, Default: 1) Epsilon greedy start.
- `--epsilon_end`: (Type: int, Default: 0.1) Epsilon greedy end.
- `--theta_start`: (Type: int, Default: 5) Theta, the death horizon start.
- `--theta_end`: (Type: int, Default: 25) Theta, the death horizon end.
- `--train_e`: (Type: bool, Default: False) Train epsilon.

*Note: Uncomment the relevant lines in your script before using.*

## Run the inference 
in the file DRL_model_SurvivalFocused.py their is the function : test_policy(env, ff_net_trained, save_vid=True) for this purpuse.

## Results
[![Video SurvFocused Cartpole](results/videos/survFocused.mp4)](results/videos/survFocused.mp4)
<img src="results/report/surv-focused_vs_rnd.png" alt="Cartpole survvsrnd" width="50%">

# Part 2 : DQN which learns with Forward Forward algorithm
This adapts the FF to the well knwon DQN algorithm. It also proposes a way to solve a regression task with FF.

## Run the training phase
Run the Regression FF : `python FF_regression.py`

This scripts comes with the follwing parsers
- `--train_ff`: *(type: bool, default: True)* If set to `True`, the model will be trained using the foward-forward (FF) method.

- `--train_bp`: *(type: bool, default: False)* If set to `True`, the model will be trained using the backpropagation (BP) method.

- `--num_epochs`: *(type: int, default: 100)* Specifies the number of training epochs for the model.

Run the forwrad-forward DQN : `python DRL_model_Qlearning.py`

## Run the inference 
FF_DRL/code/Simple_DQN/DRL_model_Qlearning.py look for the test_policy


## Results


<img src="results/FF_Qlearning/10_training_DQL_results.png" alt="Cartpole FF" width="50%">

![Breakout with BP](results/videos/500steps.gif)

[![Video DQN Cartpole FF](results/videos/500steps.mp4)](results/videos/500steps.mp4)

# Part 3 : Advanced DQN
This is an attempt of solving mroe complex env like breakout. This kind of env are interessting because they are dealing with images, and requires CNN to be efficient. 

## Run the training phase
Train the Forward Forward with CNN : `python FF_CNN.py`

Train Breakout DQN with BP : `python train_backpropagation.py`

Train Breakout DQN with FF : `python train_forwardforward.py`


## Run the inference 
Test Breakout DQN with BP : `python test_backpropagation.py`

Test Breakout DQN with FF : `python test_forwradforward.py`

## Results

The DQN with BP performs good : 


![Breakout with BP](results/videos/gif_breakout.gif)


[![Video Statndard DQN Breakout](results/videos/DQN_breakout_bp.mp4)](results/videos/DQN_breakout_bp.mp4)




The FF early results are promissing, This part must be explorered a bit more :


<img src="results/report/FF_DQN_Advanced_2ndFE.png" alt="Cartpole res" width="50%">

[![Video Forward-Forward DQN Breakout](results/videos/FFDQN_breakout.mp4)](results/videos/FFDQN_breakout.mp4)


# 🚀 Forward-Forward RL: Backprop-Free Deep Reinforcement Learning

> **EPFL Project · Robotics & Data Science**  
> **Author :** Robin Junod · **Supervisor :** Giulio Romanelli

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/pytorch-2.2-orange)](https://pytorch.org/)  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*Re-imagining Deep Q-Learning without gradients—powered by Geoffrey Hinton’s **Forward-Forward** Algorithm.*

---

## ✨ Why this repo might grab your attention

- **Gradient-free training** for RL — no back-prop
- Three **self-contained experiments**  
  1. **Survival-Focused Algo (Vanilla CartPole)**  
  2. **DDQN on CartPole (Vector-World)**  
  3. **DDQN on Breakout (Image/CNN World)**
- End-to-end Atari self playing agent (Breakout 🕹) + classic control (CartPole)
- Clean, reproducible code with  **PyTorch**, **Gymnasium**, <abbr title="Weights & Biases">wandb</abbr> logging

---

## 🎥 Results Demo Breakout

![Breakout with BP](results/videos/500steps.gif)
<p align="center">
  <img src="results/videos/500steps.gif" alt="Cartpole" width="65%">
</p>

<p align="center">
  <img src="results/videos/gif_breakout.gif" alt="Breakout BP" width="65%">
</p>


---

## 🗺️ Repo Tour

| Folder | Experiment |
|--------|------------|
| **code/Survival_Focused** | **1 · Survival-Focused Algo (Vanilla CartPole)** |
| **code/Simple_DQN**       | **2 · DDQN on CartPole (Vector-World)** |
| **code/Advanced_DQN**     | **3 · DDQN on Breakout (Image/CNN World)** |
| **results/**              | 📊 Plots, 📽 GIFs & videos |

---

## ⚡ Quick-Start

```bash
# 1 — clone
git clone https://github.com/<your-username>/forward-forward-rl.git
cd forward-forward-rl

# 2 — create *identical* environment
conda env create -f environment.yml
conda activate ff_rl

## Detailed Results
