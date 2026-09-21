import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state):
        return self.network(state)

class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, 4)

    def forward(self, state):
        features = self.features(state)

        value = self.value(features)
        advantage = self.advantage(features)

        return value + advantage - advantage.mean(dim=-1, keepdim=True)

checkpoints = [
    ("models/best_model_validation.pth", DQN),
    ("models/dueling_best_model_validation.pth", DuelingDQN),
    ("models/per_best_model_validation.pth", DuelingDQN)
]

env = gym.make("LunarLander-v3")

for checkpoint, model_class in checkpoints:
    model = model_class()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    rewards = []

    for seed in range(100):
        state, info = env.reset(seed=seed)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                q_values = model(state_tensor)

            action = torch.argmax(q_values).item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    rewards = np.array(rewards)

    print(f"\n--- {checkpoint} ---")
    print(f"Mean: {rewards.mean():.1f}")
    print(f"Std: {rewards.std():.1f}")
    print(f"Best: {rewards.max():.1f}")
    print(f"Worst: {rewards.min():.1f}")
    print(f">= 200: {(rewards >= 200).sum()}/100")

env.close()