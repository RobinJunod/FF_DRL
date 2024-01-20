# FF_DRL
This Repo is an attempt to solve gym env using the forwardforward leanring metho from g.hinton. Autors Robin Junod, supervised by Giulio Romanelli. This is a semester project at the EPFL, Master (robotic/datascience).

It can be devided in 3 parts

1 survival focused , cartpole 
2 DQN , cartpole 
3 DQN with CNN, breakout

## Installation and depedencies
packages needed: pytorch, sklearn, pandas, numpy, gymnasium, gym atari (New version), matplotlib

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
[![Video Thumbnail](results/videos/survFocused.mp4)](results/videos/survFocused.mp4)
![FF vs Rnd](results/report/surv-focused_vs_rnd.png)


# Part 2 : simple DQN
Adapt the FF to the well knwon DQN algorithm. This task contains a new algorithm to solve regression task with Forward-Forward. 

## Run the training phase
Run the fowrad-forward DQN : `python DRL_model_Qlearning.py`
## Run the inference 

## Results



# Part 3 : advanced DQN
This is an attempt of solving mroe complex env like breakout. This kind of env are interessting because they are dealing with images, and requires CNN to be efficient. 
## Run the training phase
## Run the inference 
## Results