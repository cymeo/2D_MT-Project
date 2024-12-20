import numpy as np 
import math 
import matplotlib.pyplot as plt
import pandas as pd



log_data = pd.read_csv("monitor_logs/monitor.csv", skiprows=1)
# Extract episode rewards and timesteps
timesteps = log_data["l"].cumsum()  # Cumulative timesteps
episode_rewards = log_data["r"]  # Rewards
window_size = 50
smoothed_rewards = episode_rewards.rolling(window=window_size, min_periods=1).mean()  
  
  
plt.figure(figsize=(10, 6))
plt.plot(timesteps, episode_rewards, label="Episode Reward")
plt.plot(timesteps, smoothed_rewards, label=f"Smoothed Reward (Window={window_size})", color='orange')
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Learning Curve")
plt.legend()
plt.grid()
plt.show()
