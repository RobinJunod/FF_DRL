#!/bin/bash

# Define the Python script to execute
python_script="your_script.py"

# Define an array of different sets of command-line arguments
args=(
    "--num_episodes 1000 --theta_start 2 --theta_decay 1.01 --theta_end 30"
    "--num_episodes 2000 --theta_start 2 --theta_decay 1.01 --theta_end 30"
    "--num_episodes 3000 --theta_start 2 --theta_decay 1.01 --theta_end 30"
    "--num_episodes 1000 --theta_start 5 --theta_decay 1.01 --theta_end 10"
    "--num_episodes 1000 --theta_start 20 --theta_decay 0.995 --theta_end 5"
)

# Loop through the array of arguments and execute the Python script
for arg in "${args[@]}"; do
    python "$python_script" $arg
done