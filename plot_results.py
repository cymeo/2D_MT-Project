import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter1d

def plot_monitor_data(file_path):
    # Load the monitor CSV file
    data = pd.read_csv(file_path, skiprows=1)  # Skip the first row (commented metadata)

    # Extract rewards and episode lengths (steps)
    rewards = data['r']
    episode_lengths = data['l']

    # Calculate the epochs
    epochs = np.arange(len(rewards))

    # filter for episode lengths
    X = epochs.reshape(-1, 1)  # Reshape for sklearn
    y_epF = uniform_filter1d(episode_lengths,20)
    
    # filter for rewards
    y_rewards_f = uniform_filter1d(rewards,20)
    

    # Create the subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot rewards
    axs[0].scatter(epochs, rewards, label='Rewards', color='blue')
    axs[0].plot(epochs, y_rewards_f, label='filtered', color='red', linewidth=2)
    axs[0].set_title('Rewards per Epoch')
    axs[0].set_ylabel('Rewards')
    axs[0].legend()
    axs[0].grid()

    # Plot steps (episode lengths) with linear regression
    axs[1].scatter(epochs, episode_lengths, label='Steps', color='green', alpha=0.7)
    axs[1].plot(epochs, y_epF, label='filtered', color='red', linewidth=2)
    
    axs[1].set_title('Steps per Epoch')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Steps')
    axs[1].legend()
    axs[1].grid()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage
plot_monitor_data('monitor_logs/succ1.csv')





