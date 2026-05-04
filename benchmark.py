import flappy_bird_gymnasium
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
model = PPO.load("flappy_ppo_v2", env=env)

scores = []
for episode in range(20):
    obs, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    scores.append(total_reward)
    print(f"Episode {episode+1}: {total_reward:.1f}")

print(f"\nAverage: {sum(scores)/len(scores):.1f}")
print(f"Best: {max(scores):.1f}")
print(f"Worst: {min(scores):.1f}")

env.close()