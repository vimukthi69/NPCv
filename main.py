import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class GridDrivingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_size=(20, 3), num_npvs=4, style="standard"):
        super(GridDrivingEnv, self).__init__()
        self.grid_size = grid_size
        self.num_npvs = num_npvs
        self.previous_row = self.grid_size[0] - 1
        self.previous_col_progress = self.grid_size[1] // 2
        self.style = style

        # Define action and observation space
        # Actions:
        # 0 = keep lane
        # 1 = overtake
        # 2 = go to rightmost lane
        self.action_space = spaces.Discrete(3)

        # Observation space: grid representation
        # 0 = empty, 1 = ego vehicle, 2 = NPV
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.grid_size[0] * self.grid_size[1],), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_row = self.grid_size[0] - 1
        self.previous_col_progress = self.grid_size[1] // 2
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
        self.ego_pos = [self.grid_size[0] - 1, self.grid_size[1] // 2]
        self.grid[self.ego_pos[0], self.ego_pos[1]] = 1  # Ego vehicle
        self.npvs = self._spawn_npvs()
        self._update_grid()
        return self.grid.flatten(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # Apply action
        self._move_ego(action)
        self._move_npvs()

        # Update grid state
        self._update_grid()

        # Check for collisions
        if self.grid[self.ego_pos[0], self.ego_pos[1]] > 1:  # Collision with NPV
            reward = -1  # Immediate penalty for collision
            terminated = True
        else:
            # Assign rewards only if no collision
            reward = self._calculate_reward()

        # Episode termination condition (e.g., reaching the top row)
        if self.ego_pos[0] == 0:
            terminated = True

        return self.grid.flatten(), reward, terminated, truncated, {}

    def render(self):
        # Simple console printout of the grid
        print(self.grid)

    def close(self):
        pass

    def _move_ego(self, action):
        """
        Overtake logic in the paper
        Step 1: Check if the left lane is clear.
        Step 2: Move into the left lane.
        Step 3: Continue moving forward until the slower vehicle in the original lane is passed
        :param action:
        :return: next position
        """
        row, col = self.ego_pos

        if action == 0:  # Keep Lane
            if not self._is_vehicle_ahead():
                row = max(0, row - 1)  # Accelerate if no vehicle ahead

        elif action == 1:  # Overtake
            if col > 0 and not self._is_vehicle_in_lane(col - 1):  # Check if left lane is clear
                col -= 1  # Move left
                if not self._is_vehicle_ahead():  # Check if there's space to move forward
                    row = max(0, row - 1)

        elif action == 2:  # Go to Rightmost Lane
            if col < self.grid_size[1] - 1 and not self._is_vehicle_in_lane(col + 1):  # Check right lane
                col += 1

        self.ego_pos = [row, col]

    def _move_npvs(self):
        new_npvs = []
        for row, col in self.npvs:
            # Move the NPV down or keep in the same position
            if row < self.grid_size[0] - 1:
                row += 1
            else:
                # Respawn the NPV at the top
                row = 0
                col = np.random.randint(0, self.grid_size[1])

            new_npvs.append([row, col])
        self.npvs = new_npvs

    def _calculate_reward(self):
        reward = 0

        # Reward weights based on driving style
        if self.style == "comfort":
            lane_reward_weight = 0.2
            speed_reward_weight = -0.1
            collision_penalty = -2
        elif self.style == "aggressive":
            lane_reward_weight = 0.05
            speed_reward_weight = 0.2
            collision_penalty = -0.5
        else:  # Standard
            lane_reward_weight = 0.1
            speed_reward_weight = 0.1
            collision_penalty = -1

        # Reward for moving closer to the rightmost lane
        rightmost_lane = self.grid_size[1] - 1
        col_progress = rightmost_lane - self.ego_pos[1]  # Distance to the rightmost lane
        if col_progress < self.previous_col_progress:  # Ego moves closer to the rightmost lane
            reward += 0.05  # Reward for progress toward the rightmost lane
        if self.ego_pos[1] == rightmost_lane:  # Ego is in the rightmost lane
            reward += lane_reward_weight

        # Penalty for collisions
        if self.grid[self.ego_pos[0], self.ego_pos[1]] > 1:
            return collision_penalty

        # High-speed reward/penalty
        forward_progress = self.previous_row - self.ego_pos[0]
        if forward_progress > 0:  # Positive progress
            if self.style == "comfort":
                reward += speed_reward_weight  # Penalize high speeds
            else:
                reward += speed_reward_weight  # Reward for progress in standard/aggressive styles
        elif forward_progress == 0:  # No movement
            if self.style != "comfort":  # Penalize idling only for non-comfort styles
                reward -= 0.1

        # Update the previous_row for the next step
        self.previous_row = self.ego_pos[0]
        self.previous_col_progress = col_progress

        # Normalize reward by maximum possible magnitude
        max_reward = max(abs(lane_reward_weight), abs(collision_penalty), abs(speed_reward_weight))
        scaled_reward = reward / max_reward  # Scale by max reward magnitude

        return scaled_reward

    def _spawn_npvs(self):
        npvs = []
        for _ in range(self.num_npvs):
            valid_position = False
            while not valid_position:
                row = np.random.randint(0, self.grid_size[0] - 2)  # Exclude the last row (ego's starting row)
                col = np.random.randint(0, self.grid_size[1])  # Random lane

                # Avoid horizontal clusters of NPVs
                horizontal_cluster = any(
                    [row, c] in npvs for c in range(max(0, col - 1), min(self.grid_size[1], col + 2))
                )

                # Prevent NPVs from spawning in the ego's column when ego is in the last two rows
                ego_in_last_rows = self.ego_pos[0] in [self.grid_size[0] - 1, self.grid_size[0] - 2,
                                                       self.grid_size[0] - 3]
                block_ego_column = (col == self.ego_pos[1] and ego_in_last_rows)

                # Validate position
                if not horizontal_cluster and not block_ego_column:
                    valid_position = True
                    npvs.append([row, col])

        return npvs

    def _update_grid(self):
        # Update grid with current positions of ego vehicle and NPVs
        self.grid.fill(0)
        self.grid[self.ego_pos[0], self.ego_pos[1]] = 1  # Marking ego vehicle
        for npv in self.npvs:
            self.grid[npv[0], npv[1]] = 2  # Marking NVPs

    # helper functions
    def _is_vehicle_ahead(self):
        row, col = self.ego_pos
        for r in range(row - 1, max(-1, row - 3), -1):  # Look up to 2 cells ahead
            if self.grid[r, col] == 2:  # Check for NPV
                return True
        return False

    def _is_vehicle_in_lane(self, lane):
        row, col = self.ego_pos
        # Check the lane for vehicles in the next few rows
        for r in range(row - 1, max(-1, row - 5), -1):  # Look ahead up to 3 cells
            if self.grid[r, lane] == 2:
                return True
        return False


# env = GridDrivingEnv()
#
# # Reset the environment
# obs, _ = env.reset()
# env.render()
# total_reward = 0
# # Perform a few random steps
# for _ in range(10):
#     action = env.action_space.sample()  # Choose a random action
#     obs, reward, terminated, truncated, info = env.step(action)
#     total_reward += reward
#     env.render()  # Print the grid for visualization
#     print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
#     if terminated or truncated:
#         print("Episode finished!")
#         break
