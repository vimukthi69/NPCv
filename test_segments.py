import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


'''
model = PPO("MlpPolicy", env, verbose=1)

# Train the model for a defined number of timesteps
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_grid_driving")
'''

# ----------------------------------------------------------------------
def render_grid(grid):
    plt.imshow(grid, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()


for _ in range(100):
    action = env.action_space.sample()  # Choose a random action
    obs, reward, terminated, truncated, info = env.step(action)
    render_grid(env.grid)
    if terminated or truncated:
        break

# ----------------------------------------------------------------------
# Reset the environment
obs, _ = env.reset()
env.render()

# Perform a few random steps
for _ in range(10):
    action = env.action_space.sample()  # Choose a random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Print the grid for visualization
    print(f"Action: {action}, Reward: {reward}")
    if terminated or truncated:
        print("Episode finished!")
        break
