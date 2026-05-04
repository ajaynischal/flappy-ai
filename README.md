# flappy-ai

Training an AI to play Flappy Bird using reinforcement learning. No human gameplay data, no hardcoded rules — the agent learns entirely through trial and error.

## How it works

The agent observes the game state (bird height, velocity, distance to next pipe), picks an action (flap or don't flap), and receives a reward for staying alive or passing a pipe. It dies, resets, and tries again — millions of times — gradually getting better. The algorithm handling this is PPO (Proximal Policy Optimization), which adjusts the neural network weights after each batch of gameplay.

## Setup

```bash
pip install flappy-bird-gymnasium stable-baselines3 gymnasium pygame
```

## Files

| File | What it does |
|---|---|
| `train.py` | Initial training run, 1M steps |
| `train_v2.py` | Tuned hyperparameters, 5M steps |
| `watch.py` | Loads a saved model and renders it playing live |
| `benchmark.py` | Runs 20 episodes and prints average, best, and worst scores |
| `flappy_ppo.zip` | Saved model weights from v1 |
| `flappy_ppo_v2.zip` | Saved model weights from v2 |

## Usage

Train from scratch:
```bash
python train.py
```

Continue training from a saved model:
```bash
python train_v2.py
```

Watch it play:
```bash
python watch.py
```

Benchmark a model over 20 episodes:
```bash
python benchmark.py
```

## Hyperparameters (v2)

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | 1e-3 | Faster early learning than the default |
| `batch_size` | 128 | More stable gradient updates |
| `n_epochs` | 10 | Reuses each batch more before discarding |
| `gamma` | 0.99 | Thinks roughly 100 steps ahead |
| `ent_coef` | 0.01 | Keeps some exploration, stops early convergence |
| `net_arch` | [128, 128] | Bigger network than the default [64, 64] |

## Stack

- [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://gymnasium.farama.org/)
