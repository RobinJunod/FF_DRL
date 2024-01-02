"""
This part is made to be used with cells execution

"""

#%%

def plot_comp_negTol():
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the CSV files
    file1 = "../../results/SF_experiments/reward_config_10000_3000_2_10.csv"
    file2 = "../../results/SF_experiments/reward_config_10000_3000_2_25.csv"
    file3 = "../../results/SF_experiments/reward_config_10000_3000_2_30.csv"
    file4 = "../../results/SF_experiments/reward_config_10000_3000_25_5.csv"
    file5 = "../../results/SF_experiments/3000_rndplays.csv"


    # Read the data from each file and store in a dictionary
    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    dfrnd = pd.read_csv(file5)
    # Check if dataframes are loaded


    # create mv avg and std
    window_size = 150  # or any number you choose
    df1['moving_avg'] = df1['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df2['moving_avg'] = df2['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df3['moving_avg'] = df3['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df4['moving_avg'] = df4['Reward Evolution'].rolling(window=window_size, center=True).mean()
    dfrnd['moving_avg'] = dfrnd['Reward Evolution'].rolling(window=window_size, center=True).mean()
    # Calculate the rolling standard deviation for each configuration
    window_size = 150  # Window size for the rolling calculation
    df1['std_dev'] = df1['Reward Evolution'].rolling(window=window_size, center=True).std()
    df2['std_dev'] = df2['Reward Evolution'].rolling(window=window_size, center=True).std()
    df3['std_dev'] = df3['Reward Evolution'].rolling(window=window_size, center=True).std()
    df4['std_dev'] = df4['Reward Evolution'].rolling(window=window_size, center=True).std()
    dfrnd['std_dev'] = dfrnd['Reward Evolution'].rolling(window=window_size, center=True).std()

    # plot the 4 graphs
    # Create a 2x2 grid of subplots
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    # Configure each subplot
    axs[0, 0].plot(df1['episode_R'], df1['moving_avg'], label='From 2 to 10')
    axs[0, 0].fill_between(df1['episode_R'], df1['moving_avg'] - df1['std_dev'], df1['moving_avg'] + df1['std_dev'], color='gray', alpha=0.3)
    axs[0, 0].legend(loc='upper right')  # Create a legend for this subplot

    axs[0, 1].plot(df2['episode_R'], df2['moving_avg'], label='From 2 to 25')
    axs[0, 1].fill_between(df2['episode_R'], df2['moving_avg'] - df2['std_dev'], df2['moving_avg'] + df2['std_dev'], color='gray', alpha=0.3)
    axs[0, 1].legend(loc='upper right')  # Create a legend for this subplot

    axs[1, 0].plot(df3['episode_R'], df3['moving_avg'], label='From 2 to 30')
    axs[1, 0].fill_between(df3['episode_R'], df3['moving_avg'] - df3['std_dev'], df3['moving_avg'] + df3['std_dev'], color='gray', alpha=0.3)
    axs[1, 0].legend(loc='upper right')  # Create a legend for this subplot

    axs[1, 1].plot(df4['episode_R'], df4['moving_avg'], label='From 25 to 5')
    axs[1, 1].fill_between(df4['episode_R'], df4['moving_avg'] - df4['std_dev'], df4['moving_avg'] + df4['std_dev'], color='gray', alpha=0.3)
    axs[1, 1].legend(loc='upper right')

    # Set common labels for the entire figure, adjusted for non-overlapping
    fig.text(0.5, -0.01, 'Episode', ha='center', fontsize=12)
    fig.text(-0.0, 0.5, 'Reward', va='center', rotation='vertical', fontsize=12)

    # Set a title for the entire figure
    plt.suptitle('Reward Evolution with different negative data range', fontsize=14, y=1)

    plt.tight_layout()
    plt.show()
#%%

import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files
file4 = "../../results/SF_experiments/reward_config_10000_3000_25_5.csv"
file5 = "../../results/SF_experiments/3000_rndplays.csv"

# Read the data from each file
df4 = pd.read_csv(file4)
dfrnd = pd.read_csv(file5)

# Calculate the moving average and std for file4
window_size = 150  # or any number you choose
df4['moving_avg'] = df4['Reward Evolution'].rolling(window=window_size, center=True).mean()
df4['std_dev'] = df4['Reward Evolution'].rolling(window=window_size, center=True).std()

# Calculate the moving average and std for the random file (dfrnd)
dfrnd['moving_avg'] = dfrnd['Reward Evolution'].rolling(window=window_size, center=True).mean()
dfrnd['std_dev'] = dfrnd['Reward Evolution'].rolling(window=window_size, center=True).std()

fig, ax = plt.subplots(figsize=(8, 5))
plt.style.use('seaborn-whitegrid')  # Apply a stylish background grid

# Define better shades of blue and red
blue_color = '#1f77b4'
red_color = '#ff7f0e'

# Plot the moving averages and fill the area between the
ax.plot(df4['episode_R'], df4['moving_avg'], label='Survival-Focused', linestyle='-', color=blue_color, linewidth=2)
ax.fill_between(df4['episode_R'], df4['moving_avg'] - df4['std_dev'], df4['moving_avg'] + df4['std_dev'], color=blue_color, alpha=0.2)
ax.plot(dfrnd['epsiode_R'], dfrnd['moving_avg'], label='Random actions', linestyle='-', color=red_color, linewidth=2)
ax.fill_between(dfrnd['epsiode_R'], dfrnd['moving_avg'] - dfrnd['std_dev'], dfrnd['moving_avg'] + dfrnd['std_dev'], color=red_color, alpha=0.2)
# Add legend with a fancy box
legend = ax.legend(loc='upper left', fancybox=True, framealpha=0.8, fontsize=14)
for item in legend.legendHandles:
    item.set_markersize(8)
    
# Set labels and title with larger font sizes
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Reward', fontsize=14)
ax.set_title('Survival-Focused vs Random', fontsize=16)
# Customize tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=12)
# Add a grid with a light gray color
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
# Customize the plot background color
ax.set_facecolor('#f0f0f0')
# Save or show the plot
plt.tight_layout()
plt.show()



#%% plot diff btwn 2 learning methods
import matplotlib.pyplot as plt
import pandas as pd
file1 = "../../results/SF_experiments/reward_config_10000_3000_2_25.csv"
file2 = "../../results/SF_experiments/reward_B_config_10000_3000_2_25.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
window_size = 150
df1['moving_avg'] = df1['Reward Evolution'].rolling(window=window_size, center=True).mean()
df2['moving_avg'] = df2['Reward Evolution'].rolling(window=window_size, center=True).mean()
df1['std_dev'] = df1['Reward Evolution'].rolling(window=window_size, center=True).std()
df2['std_dev'] = df2['Reward Evolution'].rolling(window=window_size, center=True).std()

#%
import matplotlib.pyplot as plt
import pandas as pd

# Read your CSV files and perform the necessary operations as mentioned in your code

# Create a figure and axis with custom style
fig, ax = plt.subplots(figsize=(8, 5))
plt.style.use('seaborn-whitegrid')  # Apply a stylish background grid

# Define better shades of blue and red
blue_color = '#1f77b4'
red_color = '#ff7f0e'

# Plot the moving averages and fill the area between them
ax.plot(df1['episode_R'], df1['moving_avg'], label='One Layer at a time', linestyle='-', color=blue_color, linewidth=2)
ax.fill_between(df1['episode_R'], df1['moving_avg'] - df1['std_dev'], df1['moving_avg'] + df1['std_dev'], color=blue_color, alpha=0.2)

ax.plot(df2['episode_R'], df2['moving_avg'], label='All Layer at once', linestyle='-', color=red_color, linewidth=2)
ax.fill_between(df2['episode_R'], df2['moving_avg'] - df2['std_dev'], df2['moving_avg'] + df2['std_dev'], color=red_color, alpha=0.2)

# Add legend with a fancy box
legend = ax.legend(loc='upper left', fancybox=True, framealpha=0.8, fontsize=14)
for item in legend.legendHandles:
    item.set_markersize(8)
    
# Set labels and title with larger font sizes
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Reward', fontsize=14)
ax.set_title('Comparison of Training Methods: one by one vs all', fontsize=16)

# Customize tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a grid with a light gray color
ax.grid(True, linestyle='--', alpha=0.5, color='gray')

# Customize the plot background color
ax.set_facecolor('#f0f0f0')

# Save or show the plot
plt.tight_layout()
plt.show()


#%%

def comp_death_horizon():
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the CSV files
    file1 = "../../results/SF_experiments/reward_B_config_10000_3000_5_10.csv"
    file2 = "../../results/SF_experiments/reward_B_config_10000_3000_5_20.csv"
    file3 = "../../results/SF_experiments/reward_B_config_10000_3000_10_5.csv"
    file4 = "../../results/SF_experiments/reward_B_config_10000_3000_20_5.csv"
    file5 = "../../results/SF_experiments/reward_B_config_10000_3000_10_10.csv"
    file6 = "../../results/SF_experiments/reward_B_config_10000_3000_20_20.csv"

    # Read the data from each file and store in a dictionary
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    df5 = pd.read_csv(file5)
    df6 = pd.read_csv(file6)

        # create mv avg and std
    window_size = 150  # or any number you choose
    df1['moving_avg'] = df1['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df2['moving_avg'] = df2['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df3['moving_avg'] = df3['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df4['moving_avg'] = df4['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df5['moving_avg'] = df5['Reward Evolution'].rolling(window=window_size, center=True).mean()
    df6['moving_avg'] = df6['Reward Evolution'].rolling(window=window_size, center=True).mean()
    # Calculate the rolling standard deviation for each configuration
    window_size = 150  # Window size for the rolling calculation
    df1['std_dev'] = df1['Reward Evolution'].rolling(window=window_size, center=True).std()
    df2['std_dev'] = df2['Reward Evolution'].rolling(window=window_size, center=True).std()
    df3['std_dev'] = df3['Reward Evolution'].rolling(window=window_size, center=True).std()
    df4['std_dev'] = df4['Reward Evolution'].rolling(window=window_size, center=True).std()
    df5['std_dev'] = df5['Reward Evolution'].rolling(window=window_size, center=True).std()
    df6['std_dev'] = df6['Reward Evolution'].rolling(window=window_size, center=True).std()

    # Create 3 subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Define better shades of blue and red
    blue_color = '#1f77b4'
    red_color = '#ff7f0e'

    # First subplot: Compare 5 to 10 and 10 to 5
    axs[0].plot(df1['episode_R'], df1['moving_avg'], label='From 5 to 10', linestyle='-', color=blue_color, linewidth=2)
    axs[0].fill_between(df1['episode_R'], df1['moving_avg'] - df1['std_dev'], df1['moving_avg'] + df1['std_dev'], color=blue_color, alpha=0.2)
    axs[0].plot(df2['episode_R'], df2['moving_avg'], label='From 10 to 5', linestyle='-', color='green', linewidth=2)
    axs[0].fill_between(df2['episode_R'], df2['moving_avg'] - df2['std_dev'], df2['moving_avg'] + df2['std_dev'], color='green', alpha=0.2)
    axs[0].legend(loc='upper left', fancybox=True, framealpha=0.8, fontsize=12)
    axs[0].axhline(y=100, color='black', linestyle='--')  # Add a black line at y=100

    # Second subplot: Compare 5 to 20 and 20 to 5
    axs[1].plot(df3['episode_R'], df3['moving_avg'], label='From 5 to 20', linestyle='-', color=red_color, linewidth=2)
    axs[1].fill_between(df3['episode_R'], df3['moving_avg'] - df3['std_dev'], df3['moving_avg'] + df3['std_dev'], color=red_color, alpha=0.2)
    axs[1].plot(df4['episode_R'], df4['moving_avg'], label='From 20 to 5', linestyle='-', color='purple', linewidth=2)
    axs[1].fill_between(df4['episode_R'], df4['moving_avg'] - df4['std_dev'], df4['moving_avg'] + df4['std_dev'], color='purple', alpha=0.2)
    axs[1].legend(loc='upper left', fancybox=True, framealpha=0.8, fontsize=12)
    axs[1].axhline(y=100, color='black', linestyle='--')  # Add a black line at y=100

    # Third subplot: Compare 10 to 10 and 20 to 20
    axs[2].plot(df5['episode_R'], df5['moving_avg'], label='Always 10', linestyle='-', color='orange', linewidth=2)
    axs[2].fill_between(df5['episode_R'], df5['moving_avg'] - df5['std_dev'], df5['moving_avg'] + df5['std_dev'], color='orange', alpha=0.2)
    axs[2].plot(df6['episode_R'], df6['moving_avg'], label='Always 20', linestyle='-', color='brown', linewidth=2)
    axs[2].fill_between(df6['episode_R'], df6['moving_avg'] - df6['std_dev'], df6['moving_avg'] + df6['std_dev'], color='brown', alpha=0.2)
    axs[2].legend(loc='upper left', fancybox=True, framealpha=0.8, fontsize=12)
    axs[2].axhline(y=100, color='black', linestyle='--')  # Add a black line at y=100

    # Set common labels for the entire figure
    fig.text(0.5, -0.04, 'Episode', ha='center', fontsize=14)
    fig.text(-0.04, 0.5, 'Reward', va='center', rotation='vertical', fontsize=14)

    # Customize tick label font sizes for all subplots
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Add a grid with a light gray color to all subplots
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')

    # Customize the plot background color for all subplots
    for ax in axs:
        ax.set_facecolor('#f0f0f0')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Add a global title to the entire figure
    fig.suptitle('Comparison of Negative Data Ranges', fontsize=16, y=1.02)

    # Show the plot
    plt.show()

#%%
