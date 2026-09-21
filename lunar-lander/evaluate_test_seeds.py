import json
import datetime as dt

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
M = "models/"   # run from the lunar-lander project folder


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


def load(fname):
    sd = torch.load(M + fname, map_location="cpu")
    net = DuelingDQN() if "features.0.weight" in sd else PlainDQN()
    net.load_state_dict(sd)
    net.eval()
    return net


def rollout(net, env, seed):
    state, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        with torch.no_grad():
            a = int(torch.argmax(net(torch.tensor(state, dtype=torch.float32))))
        state, r, term, trunc, _ = env.step(a)
        total += r
        steps += 1
        if term or trunc:
            u = env.unwrapped
            if trunc:
                outcome = "timeout"
            elif u.game_over or abs(state[0]) >= 1.0:
                outcome = "crash"
            else:
                outcome = "landed"
            return total, steps, outcome


def evaluate(net, seeds):
    env = gym.make("LunarLander-v3")
    out = [rollout(net, env, s) for s in seeds]
    env.close()
    R = np.array([o[0] for o in out])
    S = np.array([o[1] for o in out])
    oc = [o[2] for o in out]
    return {
        "mean": float(R.mean()),
        "std": float(R.std()),
        "median": float(np.median(R)),
        "min": float(R.min()),
        "pct_ge_200": float(100 * (R >= 200).mean()),
        "landed": oc.count("landed"),
        "crash": oc.count("crash"),
        "timeout": oc.count("timeout"),
        "mean_len": float(S.mean()),
        "rewards": R.tolist(),
    }


MODELS = [
    ("Baseline best", "best_model_validation.pth"),
    ("Baseline final", "huber_100k_validation_3000.pth"),
    ("Dueling best (old cfg)", "dueling_best_model_validation.pth"),
    ("Dueling final (old cfg)", "dueling_huber_100k_validation_3000.pth"),
    ("PER best", "per_best_model_validation.pth"),
    ("PER final", "per_huber_100k_validation_3000.pth"),
]

VAL_SEEDS = list(range(10000, 10020))   # the seeds used for checkpoint selection
TEST_SEEDS = list(range(20000, 20100)) + list(range(30000, 30100))  # never seen in training (42..3041) or selection

if __name__ == "__main__":
    results = {}
    for label, f in MODELS:
        net = load(f)
        val = evaluate(net, VAL_SEEDS)
        test = evaluate(net, TEST_SEEDS)
        results[label] = {"file": f, "val": val, "test": test}
        print(
            f"{label:24s} | val(20) {val['mean']:6.1f} | "
            f"test({len(TEST_SEEDS)}) mean {test['mean']:6.1f} ± {test['std']:5.1f}  "
            f"median {test['median']:6.1f}  min {test['min']:7.1f}  "
            f">=200 {test['pct_ge_200']:3.0f}%  "
            f"landed {test['landed']:3d} crash {test['crash']:2d} timeout {test['timeout']:2d}  "
            f"len {test['mean_len']:5.0f}",
            flush=True,
        )

    with open("logs/test_eval_results.json", "w") as fh:
        json.dump(results, fh)
