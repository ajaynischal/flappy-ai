import flappy_bird_gymnasium
import gymnasium as gym
from stable_baselines3 import PPO
import time 

# load the environment with rendering
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

# load the trained model
model = PPO.load("flappy_ppo_v2", env=env)

# run for 5 episodes
for episode in range(5):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        time.sleep(0.01)  # Slight delay so it doesn't run too fast to see
    
    print(f"Episode {episode + 1} — Score: {total_reward:.1f}")

env.close()