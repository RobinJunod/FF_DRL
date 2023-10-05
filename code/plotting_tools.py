#%%
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# From a list of tuple 2D plot a simple graph
def linear_graph(tuples_list, x_label, y_label, title):
    # Extract x and y values from the list of tuples
    x_values, y_values = zip(*tuples_list)
    # Create a line plot
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    # Add labels and a title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # Show the plot
    plt.show()
    
# Plot moving average from a list of tuples
def moving_average(tuples_list, x_axis_name='Time', y_axsis_name='Reward', title='Moving average plot', window_size=3):
    # Check of good inputs
    if not isinstance(x_axis_name, str) or not isinstance(y_axsis_name, str) or not isinstance(title, str) :
        raise ValueError("Input x_name, y_name, title must be a string")
    if not isinstance(tuples_list, list):
        raise ValueError("Input must be a list")
    for item in tuples_list:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("Each element in the list must be a 2D tuple (tuple with two elements)")
    
    # Extract x and y values into separate lists
    x_values, y_values = zip(*tuples_list)

    # Calculate the moving average using numpy
    moving_avg = np.convolve(y_values, np.ones(window_size)/window_size, mode='valid')

    # Create a new list of x values for the moving average
    x_avg = x_values[(window_size-1)//2 : -(window_size-1)//2]

    # Plot the original data points
    plt.scatter(x_values, y_values, label='Original Data', marker='o')

    # Plot the moving average
    plt.plot(x_avg, moving_avg, label=f'Moving Average (Window Size {window_size})', color='red')

    # Add labels and legend
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axsis_name)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.title(title)
    plt.show()
    
def tuple_list_from_csv(filename):
    """plot moving average and std of a pandas dataframe
    Args:
        filename (_type_): _description_
    """
    
    df = pd.read_csv(filename)
    
    f_name = df.columns[0]
    s_name = df.columns[1]
    
    # Set the window size for the moving average and std calculation
    window_size = 50  # You can adjust this value

    # Calculate the moving average using Pandas rolling function
    df['MovingAverage'] = df[s_name].rolling(window=window_size, center=True).mean()

    # Calculate the standard deviation using Pandas rolling function
    df['StdDeviation'] = df[s_name].rolling(window=window_size,center=True).std()

    # Create a plot using Seaborn and Matplotlib
    plt.figure()
    sns.scatterplot(x=f_name, y=s_name, data=df, label='Raw Data', color='blue',alpha=0.3)
    sns.lineplot(x=f_name, y='MovingAverage', data=df, label='Moving Average', color='red')

    # Fill the area between MovingAverage - StdDeviation and MovingAverage + StdDeviation with a shaded region
    plt.fill_between(df[f_name], df['MovingAverage'] - df['StdDeviation'], df['MovingAverage'] + df['StdDeviation'], alpha=0.2, color='green', label='Std Deviation')

    # Customize the plot
    plt.xlabel('Episode')
    plt.ylabel(s_name)
    plt.title(f'{s_name}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    # Show the plot
    plt.show()

    

# %%
