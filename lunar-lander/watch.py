import gymnasium as gym
import torch
import torch.nn as nn
import sys

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

model = DuelingDQN()
model.load_state_dict(torch.load("models/dueling_seed3042_best.pth"))
model.eval()

seed = int(sys.argv[1])

env = gym.make("LunarLander-v3", render_mode="human")

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

print(f"Seed {seed}, Reward: {total_reward:.1f}")

env.close()