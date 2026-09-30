"""
Evaluate every method x training seed on the same unseen starting positions,
then summarise across seeds.

Run from the lunar-lander project folder:
    python evaluate_seeds.py

Expects checkpoints named  models/{method}_seed{seed}_best.pth  and  _final.pth
(missing files are skipped with a note, so you can run it while PER is still training).
"""
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

METHODS = ["baseline", "dueling", "per"]
TRAIN_SEEDS = [42, 3042, 6042]
CHECKPOINT = "best"          # "best" (picked by validation) or "final"
TEST_SEEDS = list(range(20000, 20100)) + list(range(30000, 30100))


class PlainDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.network(x)


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, 4)

    def forward(self, x):
        f = self.features(x)
        a = self.advantage(f)
        return self.value(f) + a - a.mean(dim=-1, keepdim=True)


def load(path):
    sd = torch.load(path, map_location="cpu")
    net = DuelingDQN() if "features.0.weight" in sd else PlainDQN()
    net.load_state_dict(sd)
    net.eval()
    return net


def run_episode(net, env, seed):
    state, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        with torch.no_grad():
            action = int(torch.argmax(net(torch.tensor(state, dtype=torch.float32))))
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            if truncated:
                outcome = "timeout"
            elif env.unwrapped.game_over or abs(state[0]) >= 1.0:
                outcome = "crash"
            else:
                outcome = "landed"
            return total, outcome


def test(net):
    env = gym.make("LunarLander-v3")
    results = [run_episode(net, env, s) for s in TEST_SEEDS]
    env.close()
    rewards = np.array([r for r, _ in results])
    outcomes = [o for _, o in results]
    return rewards, outcomes.count("crash"), outcomes.count("timeout")


if __name__ == "__main__":
    summary = {}
    print(f"Testing '{CHECKPOINT}' checkpoints on {len(TEST_SEEDS)} unseen starting positions\n")
    for method in METHODS:
        per_seed = []
        for seed in TRAIN_SEEDS:
            path = f"models/{method}_seed{seed}_{CHECKPOINT}.pth"
            try:
                net = load(path)
            except FileNotFoundError:
                print(f"  {method:8s} seed {seed:5d}   (missing: {path})")
                continue
            rewards, crashes, timeouts = test(net)
            per_seed.append({"seed": seed, "mean": float(rewards.mean()),
                             "pct_ge_200": float(100 * (rewards >= 200).mean()),
                             "crashes": crashes, "timeouts": timeouts,
                             "rewards": rewards.tolist()})
            print(f"  {method:8s} seed {seed:5d}   mean {rewards.mean():6.1f}   "
                  f">=200 {100 * (rewards >= 200).mean():3.0f}%   "
                  f"crashes {crashes:2d}   timeouts {timeouts:2d}")

        if per_seed:
            means = np.array([s["mean"] for s in per_seed])
            spread = means.std(ddof=1) if len(means) > 1 else float("nan")
            crashes = sum(s["crashes"] for s in per_seed)
            episodes = len(per_seed) * len(TEST_SEEDS)
            print(f"  {method:8s} ACROSS {len(per_seed)} SEEDS: mean {means.mean():6.1f} "
                  f"± {spread:5.1f} (std across seeds)   "
                  f"crash rate {100 * crashes / episodes:.1f}%\n")
            summary[method] = {"per_seed": per_seed,
                               "mean_across_seeds": float(means.mean()),
                               "std_across_seeds": float(spread),
                               "crash_rate_pct": 100 * crashes / episodes}

    with open(f"logs/seed_eval_{CHECKPOINT}.json", "w") as fh:
        json.dump(summary, fh)
    print(f"saved logs/seed_eval_{CHECKPOINT}.json")
