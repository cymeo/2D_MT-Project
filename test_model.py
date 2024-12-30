import gym
from stable_baselines3 import PPO
from robot import TwoD_Robot


env = TwoD_Robot()
model = PPO.load("2DRobot_500eps")

num_episodes = 10
rewards = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("done_True")
    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward}")

print(f"Average reward over {num_episodes} episodes: {sum(rewards) / num_episodes}")