#!/bin/bash

# Define the Python script to execute
python_script="DRL_model.py"

# Define an array of different sets of command-line arguments
args=(
    "--num_episodes 3000 --theta_start 25 --theta_end 5"
    "--num_episodes 3000 --theta_start 2 --theta_end 25"
    "--num_episodes 3000 --theta_start 2 --theta_end 10"
    "--num_episodes 10000 --theta_start 2 --theta_end 40"
)

# Loop through the array of arguments and execute the Python script
for arg in "${args[@]}"; do
    python "$python_script" $arg
done