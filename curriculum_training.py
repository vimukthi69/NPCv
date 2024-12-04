import numpy as np
from stable_baselines3 import PPO
from main import GridDrivingEnv


# Define environments with increasing complexity
def create_env(level):
    if level == 1:
        return GridDrivingEnv(grid_size=(20, 3), num_npvs=2, style="standard")  # Simpler environment
    elif level == 2:
        return GridDrivingEnv(grid_size=(20, 3), num_npvs=5, style="aggressive")  # Medium complexity
    else:
        return GridDrivingEnv(grid_size=(20, 3), num_npvs=8, style="aggressive")  # Complex environment


# Training process with curriculum learning
def train_with_curriculum(model, levels=3, total_timesteps=500000):
    timesteps_per_level = total_timesteps // levels  # Divide total timesteps by the number of levels

    for level in range(1, levels + 1):
        print(f"Training on level {level} with {timesteps_per_level} timesteps")

        # Create environment for the current level
        env = create_env(level)

        # Set up the model with the current environment
        model.set_env(env)

        # Train the model on the current level
        model.learn(total_timesteps=timesteps_per_level)

        # Save the model after each level
    model.save(f"trained_curriculum_models/ppo_grid_driving_aggressive_curriculum")


# Create the PPO model (with initial environment)
env = create_env(1)  # Start with the simplest environment
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, gamma=0.99, learning_rate=0.0001, verbose=1)

# Start training with curriculum learning
train_with_curriculum(model, levels=3, total_timesteps=500000)
