import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import ale_py
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Resize
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback

gym.register_envs(ale_py)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Resize & Preprocess ----------------
resize_to_cifar = Resize((32, 32), antialias=True)

def preprocess(obs):
    obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0  # [3, H, W]
    obs = resize_to_cifar(obs)
    return obs.unsqueeze(0).to(device)  # [1, 3, 32, 32]

process = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
])

# ---------------- VAE Model ----------------
latent_dim = 128

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(self.fc_decode(z))
        return out, mu, logvar

# ---------------- Load Trained VAE ----------------
VAE_model = VAE().to(device)
checkpoint = torch.load("vae_cifar100.pth", map_location=device)
VAE_model.load_state_dict(checkpoint["model_state_dict"])
VAE_model.eval()

# ---------------- Gym Wrapper ----------------
class LatentWrapper(gym.Env):
    def __init__(self, other_env, latent_dim=latent_dim):
        self.other_env = other_env
        self.latent_dim = latent_dim

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(latent_dim,), dtype=np.float32)
        self.action_space = self.other_env.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.other_env.reset(seed=seed, options=options)
        observation = process(observation).unsqueeze(0).to(device)
        _, mu, logvar = VAE_model(observation)
        z = VAE_model.reparameterize(mu, logvar)
        z = torch.tanh(z).squeeze(0).detach().cpu().numpy()
        return z, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.other_env.step(action)
        observation = process(observation).unsqueeze(0).to(device)
        _, mu, logvar = VAE_model(observation)
        z = VAE_model.reparameterize(mu, logvar)
        z = torch.tanh(z).squeeze(0).detach().cpu().numpy()
        return z, reward, terminated, truncated, info

    def close(self):
        self.other_env.close()

# ---------------- Reward Tracking Callback ----------------
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
        env = LatentWrapper(base_env)
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
        fname = f"vae_{agent_name.lower()}_{env_id.split('/')[-1].lower()}_rewards.png"
        plt.savefig(fname)
        plt.close()