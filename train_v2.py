import flappy_bird_gymnasium
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    # --- Learning rate ---
    learning_rate=1e-3,       # default 3e-4 — higher means faster early learning
    
    # --- Experience collection ---
    n_steps=2048,             # how many steps to collect before each update
    batch_size=128,           # how many samples to train on per update (default 64)
    n_epochs=10,              # how many times to reuse each batch
    
    # --- Reward discounting ---
    gamma=0.99,               # how much it values future rewards (0=now, 1=far future)
    gae_lambda=0.95,          # balances bias vs variance in advantage estimation
    
    # --- PPO stability ---
    clip_range=0.2,           # how much the policy can change per update
    ent_coef=0.01,            # encourages exploration — stops it getting stuck early
    
    # --- Network size ---
    policy_kwargs=dict(
        net_arch=[128, 128]   # bigger network than default [64, 64]
    ),
    
    tensorboard_log="./logs_v2/"
)

model.learn(total_timesteps=5_000_000)
model.save("flappy_ppo_v2")
print("Done! Run benchmark.py with flappy_ppo_v2 to compare.")