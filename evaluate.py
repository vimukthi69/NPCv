from stable_baselines3 import PPO
from main import GridDrivingEnv
import matplotlib.pyplot as plt

# Load the trained model
model = PPO.load("trained_models/ppo_grid_driving_standard_1")

# Define the testing environment
env = GridDrivingEnv(style="standard")


# Helper function to render the grid
def render_grid(grid):
    plt.imshow(grid, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()


# Variables to track collisions and episodes
num_runs = 20
collisions = 0

# Collision penalties for each driving style
collision_penalties = [-1, -2, -0.5]

# Run multiple episodes
for episode in range(num_runs):
    print("Episode : ", episode+1)
    obs, _ = env.reset()
    episode_collided = False  # Track if a collision occurs in the episode

    while True:
        action, _states = model.predict(obs)  # Get action from the trained model
        obs, reward, terminated, truncated, info = env.step(action)
        render_grid(env.grid)

        # Check for collision based on reward value
        if reward in collision_penalties:  # Reward matches a collision penalty
            print("collided")
            episode_collided = True

        if terminated or truncated:  # End of the episode
            if episode_collided:
                collisions += 1  # Increment collision count for this episode
            break

# Calculate collision rate as a percentage
collision_rate = (collisions / num_runs) * 100
print(f"Collision Rate: {collision_rate:.2f}%")
