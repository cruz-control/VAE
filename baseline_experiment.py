import ale_py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Resize

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback

gym.register_envs(ale_py)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Preprocessing ----------------
resize_to_cifar = Resize((32, 32), antialias=True)

process = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
])

# ---------------- Gym Wrapper ----------------
class Wrapper(gym.Env):
    def __init__(self, other_env):
        self.other_env = other_env
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(32, 32, 3), dtype=np.float32)
        self.action_space = self.other_env.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.other_env.reset(seed=seed, options=options)
        observation = process(observation).unsqueeze(0).to(device).squeeze(0).permute(1, 2, 0)
        return observation.detach().cpu().numpy(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.other_env.step(action)
        observation = process(observation).unsqueeze(0).to(device).squeeze(0).permute(1, 2, 0)
        return observation.detach().cpu().numpy(), reward, terminated, truncated, info

# ---------------- Callback ----------------
class RewardTrackerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0.0

    def _on_step(self):
        self.current_rewards += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0.0
        return True

# ---------------- Agent Class Map ----------------
agents = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}

envs = [
    "ALE/Pong-v5",
    "ALE/MsPacman-v5",
    "ALE/Asteroids-v5",
    "ALE/Hangman-v5"
]

# ---------------- Training Loop ----------------
for env_id in envs:
    for agent_name, agent_class in agents.items():
        print(f"\nTraining {agent_name} on {env_id}...")
        base_env = gym.make(env_id, obs_type="rgb")
        env = Wrapper(base_env)
        callback = RewardTrackerCallback()

        model = agent_class("MlpPolicy", env, verbose=0, device=device)
        model.learn(total_timesteps=40000, callback=callback, progress_bar=True)

        # Save Plot
        plt.figure()
        plt.plot(callback.episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{agent_name} on {env_id}")
        plt.grid(True)
        plt.legend()
        fname = f"{agent_name.lower()}_{env_id.split('/')[-1].lower()}_rewards.png"
        plt.savefig(fname)
        plt.close()
