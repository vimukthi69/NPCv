from stable_baselines3 import PPO
from main import GridDrivingEnv

model = PPO.load("ppo_grid_driving")
env = GridDrivingEnv()

num_episodes = 50
total_rewards = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    total_rewards.append(episode_reward)

average_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {average_reward}")
