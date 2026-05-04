# Import the flappy bird gymnasium library
import flappy_bird_gymnasium
# Import the gymnasium library
import gymnasium as gym
# Import the PPO agent from the stable baselines3 library
from stable_baselines3 import PPO

# Create the environment
env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

# Train for 1 million steps
model.learn(total_timesteps=3_000_000)

# Save the trained model
model.save("flappy_ppo")
print("Training done! Model saved.")
