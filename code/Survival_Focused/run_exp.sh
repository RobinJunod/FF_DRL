#!/bin/bash

# Define the Python script to execute
python_script="DRL_model_SurvivalFocused.py"

# Define an array of different sets of command-line arguments
args=(
    "--num_episodes 3000 --theta_start 5 --theta_end 10"
    "--num_episodes 3000 --theta_start 5 --theta_end 20"
    "--num_episodes 3000 --theta_start 10 --theta_end 5"
    "--num_episodes 3000 --theta_start 20 --theta_end 5"
)

# Loop through the array of arguments and execute the Python script
for arg in "${args[@]}"; do
    python "$python_script" $arg
done