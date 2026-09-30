# LunarLander Deep Reinforcement Learning

Three Deep Q-Network variants trained on Gymnasium's `LunarLander-v3`, each with three training seeds, and compared on 200 starting situations that none of the networks saw during training or checkpoint selection.

## Overview

| Variant | Script | What it adds |
|---|---|---|
| **Baseline DQN** | `dqn.py` | Standard DQN: replay buffer, target network, ε-greedy exploration |
| **Dueling DQN** | `dueling_dqn.py` | Splits the output into a state value V(s) and action advantages A(s,a) |
| **Dueling + PER** | `per_dqn.py` | Dueling network plus Prioritized Experience Replay |

Each variant adds exactly one component to the one before it. Everything else is identical, so any difference in results comes from that one component.

**Main results** (details in [Results](#results)):

- All nine trained networks solve the task: a mean return of at least 200 on unseen starts.
- **No variant is significantly better than the others.** The spread between training seeds of the same method is larger than the gap between methods.
- **Checkpoint selection matters more than the method.** Networks picked by validation beat the final networks by about 30 points on average and crash far less often.

## Environment

`LunarLander-v3` (Gymnasium, Box2D physics). The agent has to land a lander on a pad between two flags.

- **State**: 8 numbers. Position (x, y), velocity (vx, vy), angle, angular velocity, and whether each leg touches the ground.
- **Actions**: 4 discrete ones. Do nothing, fire the left engine, fire the main engine, fire the right engine.
- **Reward**:
  - shaped by distance to the pad, speed and tilt
  - +10 for each leg touching the ground
  - −0.3 per frame of main engine and −0.03 per frame of side engine
  - +100 for coming to rest, −100 for a crash
- **Episode end**: landing, crashing, flying off-screen, or a limit of 1000 steps.
- **Solved**: a mean return of at least 200.

The seed passed to `env.reset(seed=...)` fixes the terrain around the pad and the random push the lander gets at the start. The same seed with the same network always produces the same flight.

## Methods

### Shared settings (all three variants)

| Setting | Value |
|---|---|
| Network | MLP 8 → 128 → 128 → 4, ReLU |
| Target | Vanilla DQN: `r + γ · max_a Q_target(s', a)` |
| Loss | Huber (`SmoothL1Loss`) |
| Optimizer | Adam, learning rate 5e-4 |
| Discount γ | 0.99 |
| Replay buffer | 100,000 transitions |
| Batch size | 64, one gradient step per environment step |
| Warmup | Learning starts after 1,000 transitions |
| Exploration | ε from 1.0 to 0.05, multiplied by 0.995 per episode |
| Target network | Hard copy every 10 episodes |
| Training length | 3,000 episodes |
| Validation | Every 100 episodes on 100 fixed starts; the best network is saved |

The replay buffer stores `terminated`, not `terminated or truncated`, so an episode cut off by the 1000-step limit still bootstraps from the next state.

### Dueling DQN

The last layer is replaced by two heads:

```
Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
```

Subtracting the mean makes the split into V and A unique. Because every Q-value contains V, the value head gets a gradient from every transition, whichever action was taken.

### Prioritized Experience Replay

Transitions are sampled with probability proportional to `(|TD error| + 1e-6)^α`, with α = 0.6. New transitions get the current maximum priority. After each update, the sampled transitions' priorities are set to their new TD errors. Importance-sampling weights `(N · P(i))^(−β)`, normalized by the batch maximum, correct the sampling bias. β rises linearly from 0.4 to 1.0 over training.

## Evaluation protocol

Starting situations are split into three separate groups, so evaluation never reuses something the network has trained on or been selected with:

| Purpose | Seeds |
|---|---|
| Training (seed *s* uses *s* … *s*+2999) | 42–3041, 3042–6041, 6042–9041 |
| Validation (picks the best checkpoint) | 10000–10099 |
| **Test (reported results)** | **20000–20099 and 30000–30099** |

Every network is run greedily (always the highest Q-value, no exploration) on all 200 test starts. An episode's outcome is one of:

- **landed**: came to rest
- **crash**: the body touched the ground, or the lander left the screen
- **timeout**: still flying after 1000 steps

## Results

### Best checkpoints (selected on validation)

| Method | Seed 42 | Seed 3042 | Seed 6042 | **Mean ± SD across seeds** | Episodes ≥ 200 | Crash rate |
|---|---|---|---|---|---|---|
| Baseline DQN | 277.6 | 275.2 | 260.0 | **270.9 ± 9.5** | 93% | 2.3% |
| Dueling DQN | 278.0 | 283.1 | 266.5 | **275.9 ± 8.5** | 96% | 1.3% |
| Dueling + PER | 250.4 | 283.5 | 273.0 | **269.0 ± 16.9** | 95% | 4.3% |

Each cell is the mean return over 200 test episodes. Percentages are pooled over all 600 test episodes of a method.

### Final checkpoints (episode 3000)

| Method | Seed 42 | Seed 3042 | Seed 6042 | **Mean ± SD across seeds** | Episodes ≥ 200 | Crash rate |
|---|---|---|---|---|---|---|
| Baseline DQN | 284.3 | 254.7 | 186.9 | **242.0 ± 49.9** | 81% | 17.7% |
| Dueling DQN | 276.3 | 274.6 | 207.5 | **252.8 ± 39.2** | 85% | 10.2% |
| Dueling + PER | 253.9 | 222.2 | 202.5 | **226.2 ± 25.9** | 74% | 22.2% |

### Findings

1. **The three methods perform the same within noise.** No pair differs significantly: Welch's t-test gives p > 0.5 for every pair, with three seeds per method. Seeds of the same method differ by up to 33 points (PER: 250 vs 283), while the method means differ by at most 7.
2. **Checkpoint selection has a larger effect than the method.** The validated checkpoint beat the final one in 7 of 9 runs, by about 30 points on average (paired t-test, p ≈ 0.02). Final networks crashed in 10–22% of test episodes, selected ones in 1–4%. DQN's performance oscillates during training, and the last snapshot often lands in a dip: baseline seed 6042's final network scores 187, below "solved".
3. **Validation on 100 starts is a reliable guide.** For 8 of 9 runs, the validation score of the selected checkpoint was within about 10 points of its test score. The exception is PER seed 42 (validation 279, test 250).

Validation picked these episodes as best:

| Method | Seed 42 | Seed 3042 | Seed 6042 |
|---|---|---|---|
| Baseline | 2900 | 2300 | 1400 |
| Dueling | 2700 | 2900 | 1400 |
| PER | 2000 | 2300 | 1900 |

## Project structure

```
lunar-lander/
├── dqn.py                 # train baseline DQN
├── dueling_dqn.py         # train Dueling DQN
├── per_dqn.py             # train Dueling DQN + PER
├── evaluate_seeds.py      # test all methods × seeds on the 200 test starts
├── watch.py               # watch a trained network fly (opens a window)
├── requirements.txt
├── models/                # {method}_seed{seed}_best.pth and _final.pth
└── logs/                  # {method}_seed{seed}_episode_rewards / eval_rewards / losses .npy
                           # seed_eval_best.json: output of evaluate_seeds.py
```

## Setup

Tested with Python 3.13. `requirements.txt` pins the exact versions the results were produced with.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train

Each script takes the training seed as its only argument (the default is 42):

```bash
python dqn.py 42
```

Each run uses one CPU core (`torch.set_num_threads(1)`), so seeds can run in parallel:

```bash
for s in 42 3042 6042; do python -u dqn.py $s > logs/baseline_seed$s.txt 2>&1 & done
for s in 42 3042 6042; do python -u dueling_dqn.py $s > logs/dueling_seed$s.txt 2>&1 & done
wait
for s in 42 3042 6042; do python -u per_dqn.py $s > logs/per_seed$s.txt 2>&1 & done
wait
```

`-u` writes each log line immediately, so `tail -f logs/per_seed42.txt` shows live progress.

**Runtime** on an Apple Silicon MacBook Pro, three seeds in parallel:
- baseline and dueling: about 15 minutes
- PER: about 2 hours

PER is slower because this implementation rebuilds the sampling distribution over the whole buffer at every step.

### Evaluate

```bash
python evaluate_seeds.py
```

This tests every `models/{method}_seed{seed}_best.pth` on the 200 test starts, prints per-seed and across-seed results, and saves `logs/seed_eval_best.json`. Set `CHECKPOINT = "final"` at the top of the script to evaluate the final networks instead.

### Watch

```bash
python watch.py 30013
```

This opens a window and flies one episode, using the given seed as the starting situation (30013 if you leave it out). It loads `models/dueling_seed3042_best.pth`. To watch another network, change the path; baseline models also need the plain `DQN` class.

Seeds worth trying with the default network:

| Seed | Result |
|---|---|
| 30013 | Its best landing (323) |
| 30015 | A slow, careful landing (211) |
| 30086 | A crash (70) |

## Reproducibility

- Training is deterministic for a given seed on the same machine. Rerunning seed 42 reproduced earlier runs bit for bit: identical training rewards and identical final weights.
- Validation does not use the training random number generator, so changing how often or how long validation runs does not change training.
- Test results agreed within 0.5 points between an Apple Silicon Mac and an x86 Linux machine. Tiny floating-point differences occasionally flip a near-tied action choice.

## Limitations

- **Only three seeds per method.** That is enough to show the methods overlap, but too few to detect small real differences.
- **Shared hyperparameters.** PER's α and β are the defaults from the original paper, which used Atari with rewards clipped to ±1. LunarLander's rewards reach ±100, and nothing was tuned per method.
- **Vanilla DQN target**, chosen on purpose. Double DQN reduces overestimation and would probably improve all three variants.
- **The target network is synced per episode, not per step.** The interval in gradient steps therefore grows as episodes get longer.
- **PER sampling is O(N) per step.** A sum-tree would make it about as fast as the baseline.

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature 518, 529–533.
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML.
- Schaul et al. (2016). *Prioritized Experience Replay.* ICLR.
- Gymnasium, [Lunar Lander environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
