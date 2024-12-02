import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from main import GridDrivingEnv

# Lists to store metrics for manual plotting
episode_lengths = []
cumulative_rewards = []


class ManualLoggingCallback(BaseCallback):
    """
    Custom callback to log episode lengths and cumulative rewards manually.
    """
    def __init__(self, verbose=0):
        super(ManualLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check for episode info in environment logs
        if "infos" in self.locals:
            infos = self.locals["infos"]
            for info in infos:
                # Log custom metrics if they are available
                if "episode" in info.keys():
                    episode_lengths.append(info["episode"]["l"])  # Episode length
                    cumulative_rewards.append(info["episode"]["r"])  # Cumulative reward
        return True


# Initialize the environment
env = Monitor(GridDrivingEnv(style="standard"))

# Create the PPO model
model = PPO(
    "MlpPolicy",
    env,
    n_steps=4096,
    batch_size=128,
    gamma=0.997,
    ent_coef=0.1,
    learning_rate=0.0003,
    verbose=1
)

# Train the model with the custom logging callback
model.learn(total_timesteps=200000, callback=ManualLoggingCallback())

# Save the trained model
model.save("trained_models/ppo_grid_driving_standard")

# Plot Episode Lengths
plt.figure(figsize=(10, 5))
plt.plot(episode_lengths, label="Episode Length")
plt.xlabel("Episodes")
plt.ylabel("Length (steps)")
plt.title("Episode Length over Episodes")
plt.legend()
plt.show()

# Plot Cumulative Rewards
plt.figure(figsize=(10, 5))
plt.plot(cumulative_rewards, label="Cumulative Reward")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Cumulative Reward over Episodes")
plt.legend()
plt.show()
