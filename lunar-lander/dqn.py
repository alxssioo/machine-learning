import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import sys

torch.set_num_threads(1)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
METHOD = "baseline"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # take 8 inputs and produces 128 outputs
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state):
        return self.network(state)

def evaluate(model, env, num_seeds=20, start_seed=10000):
    rewards = []

    for seed in range(start_seed, start_seed + num_seeds):
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

    return np.mean(rewards)

model = DQN()

target_model = DQN()
target_model.load_state_dict(model.state_dict())
target_model.eval()

loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-4)
replay_buffer = deque(maxlen=100000)
batch_size = 64
learning_starts = 1000
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

num_episodes = 3000
env = gym.make("LunarLander-v3")
env.action_space.seed(seed)

eval_env = gym.make("LunarLander-v3")

best_eval_reward = float("-inf")
episode_rewards = []
eval_rewards = []
losses = []

for episode in range(num_episodes):
    state, info = env.reset(seed=seed + episode)
    done = False
    total_reward = 0
    episode_losses = []

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                q_values = model(state_tensor)

            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        replay_buffer.append(
            (state, action, reward, next_state, terminated)
        )

        if len(replay_buffer) >= learning_starts:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, terminals = zip(*batch)
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            terminals = torch.tensor(terminals, dtype=torch.float32)
            q_values = model(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_next_q = target_model(next_states).max(dim=1)[0]

            target_q = rewards + gamma * max_next_q * (1 - terminals)
            loss = loss_fn(current_q, target_q)
            episode_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    episode_rewards.append(total_reward)
    if episode_losses:
        losses.append(np.mean(episode_losses))
    else:
        losses.append(np.nan)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    if (episode + 1) % 100 == 0:
        eval_reward = evaluate(model, eval_env, num_seeds=100)
        eval_rewards.append((episode + 1, eval_reward))

        print(f"Evaluation: {eval_reward:.1f}")

        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            torch.save(model.state_dict(), f"models/{METHOD}_seed{seed}_best.pth")
            print("New best model saved!")

torch.save(model.state_dict(), f"models/{METHOD}_seed{seed}_final.pth")
np.save(f"logs/{METHOD}_seed{seed}_episode_rewards.npy", np.array(episode_rewards))
np.save(f"logs/{METHOD}_seed{seed}_eval_rewards.npy", np.array(eval_rewards))
np.save(f"logs/{METHOD}_seed{seed}_losses.npy", np.array(losses))

env.close()
eval_env.close()
