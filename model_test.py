from stable_baselines3 import PPO
from main import GridDrivingEnv
import matplotlib.pyplot as plt

model = PPO.load("trained_models/ppo_grid_driving_standard")
env = GridDrivingEnv()
obs, _ = env.reset()


def render_grid(grid):
    plt.imshow(grid, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()


for _ in range(100):
    action, _states = model.predict(obs)  # Get action from the trained model
    obs, reward, terminated, truncated, info = env.step(action)
    render_grid(env.grid)
    if terminated or truncated:
        break

# # Test the trained agent
# obs, _ = env.reset()
# for _ in range(20):  # Run 20 steps
#     action, _states = model.predict(obs)  # Get action from the trained model
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()  # Visualize the grid
#     print(f"Action: {action}, Reward: {reward}")
#     if terminated or truncated:
#         print("Episode finished!")
#         break
