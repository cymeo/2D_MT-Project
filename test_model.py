import gym
from stable_baselines3 import PPO
from robot import TwoD_Robot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

env = TwoD_Robot()
model = PPO.load("ppo_twod_robot")

def test_model(model): 
    num_episodes = 100
    rewards = []
    steps_per_episode =[]

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done,trunc, info = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        steps_per_episode.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    # Average reward and steps
    average_reward = np.mean(rewards)
    average_steps = np.mean(steps_per_episode)

    print(f"Average Reward over {num_episodes} episodes: {average_reward}")
    print(f"Average Steps per episode: {average_steps}")

    #3. Plot the Test Results
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), )
    
    ax[0].plot(rewards, marker='o')
    ax[0].set_title('Total Rewards Per Episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Total Reward')
    ax[0].grid()
    ax[1].plot(steps_per_episode, marker='o', color='orange')
    ax[1].set_title("Steps Per Episode")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Steps")
    ax[1].grid()

    fig.show()

    # 4. Optional: Save Results
    # Create a DataFrame
    results_df = pd.DataFrame({
        "Episode": np.arange(1, num_episodes + 1),
        "Total Reward": rewards,
        "Steps": steps_per_episode
    })
    # Save to CSV
    results_df.to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")
    
test_model(model)



