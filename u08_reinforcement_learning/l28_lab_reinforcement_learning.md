---
title: "Lab: From Value Iteration to a Deep Q-Network"
unit: "VIII"
lesson: "28"
type: lab
tags: [reinforcement-learning, mdp, bellman, value-iteration, q-learning, dqn, gymnasium, pytorch]
difficulty: intermediate
duration: "60 mins"
---

**Goal:** solve three control problems, each with less help than the last. In **Part A** the agent
is handed the rules of a small **gridworld**, so it can compute the best behaviour directly: **value
iteration** applies the **Bellman optimality** equation until it settles. In **Part B** the same
gridworld hides its rules, and **tabular Q-learning** learns the same answer from sampled moves,
exploring as it goes. In **Part C** the states are continuous, so the table becomes a neural network:
a **Deep Q-Network (DQN)** learns to land a spacecraft in Gymnasium's **LunarLander**, and you watch it
before and after training. Pairs with the concept note
[Reinforcement Learning: The Paradigm](l28_concept_reinforcement_learning.qmd).

> **Previously:** L27 -- Factor Analysis & ICA  |  **Next:** this is the final lesson of the course.

> This page is the read-only view. To run the lab, open the notebook (`l28_lab_reinforcement_learning.ipynb`) -- in Colab via the badge below, or locally. Parts A and B take seconds. Part C trains a network: about 4 minutes on a free Colab CPU. A GPU runtime (Runtime > Change runtime type > T4 GPU) is used automatically when present and is only modestly faster here. The outputs on this page came from a CPU run, which a Colab CPU reproduces exactly; a GPU or another machine follows a different training path, so its numbers differ: training usually takes 2 to 12 minutes (at most about 15, when it reaches the 300,000-step cap), and the final score on unseen episodes can land on either side of 200.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u08_reinforcement_learning/l28_lab_reinforcement_learning.ipynb)

## Scenario

A robot lives on a **3x4 grid**. One square is a wall. Two squares end the episode: a **+1** goal
(top right) and a **-1** pit just below it. Every other square costs **-0.04** to stand on, a gentle
nudge to hurry. The motor is **noisy**: the robot moves the way it intends 80% of the time and slips
90 degrees to either side 10% each; bumping a wall or an edge leaves it in place. What is the best
move in every square? We answer that three ways, each knowing less: first with the rules in hand,
then from experience alone, and finally for a lander whose state no table could hold.

## Setup

Two cells: the first installs, the second imports, picks the device, and defines the gridworld and a
drawing helper used throughout.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab ships gymnasium, torch and matplotlib; Box2D (the lander's physics) is the new piece.
%pip install -q "gymnasium[box2d]" torch matplotlib pillow
# local, in a terminal (not in the notebook):
#   uv add "gymnasium[box2d]" torch matplotlib pillow
```

```python
# Setup, cell 2 of 2 -- IMPORTS + DEVICE + THE GRIDWORLD (safe to re-run)
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"   # quiet import

import io
import copy
import random
import collections

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from PIL import Image, ImageDraw
from IPython.display import Image as GifImage, display

# use the GPU when the runtime has one
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- the gridworld (Parts A and B) ---
ROWS, COLS = 3, 4
WALL = (1, 1)
TERMINALS = {(0, 3): +1.0, (1, 3): -1.0}   # goal, pit
LIVING = -0.04       # reward R(s) of every other square
GAMMA = 1.0          # discount factor
START = (2, 0)
ACTIONS = {"N": (-1, 0), "S": (1, 0),
           "E": (0, 1), "W": (0, -1)}
PERP = {"N": ("E", "W"), "S": ("E", "W"),
        "E": ("N", "S"), "W": ("N", "S")}
states = [(r, c) for r in range(ROWS)
          for c in range(COLS) if (r, c) != WALL]

def draw_grid(values, policy, title, ax):
    """Values as colour, the policy as arrows."""
    grid = np.full((ROWS, COLS), np.nan)
    for s in states:
        grid[s] = values[s]
    ax.imshow(grid, cmap="cividis", vmin=-1, vmax=1)
    for (r, c) in [(r, c) for r in range(ROWS)
                   for c in range(COLS)]:
        s = (r, c)
        if s == WALL:
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1, fc="0.85",
                hatch="//", ec="0.5"))
            ax.text(c, r, "wall", ha="center",
                    va="center", color="0.3",
                    bbox=dict(fc="0.85", ec="none"))
            continue
        ink = "white" if values[s] < 0.1 else "black"
        if s in TERMINALS:
            ax.text(c, r, f"{TERMINALS[s]:+.0f}",
                    ha="center", va="center", color=ink,
                    fontsize=16, fontweight="bold")
            continue
        ax.text(c, r - 0.25, f"{values[s]:.3f}",
                ha="center", va="center", color=ink)
        dr, dc = ACTIONS[policy[s]]
        ax.annotate("", xy=(c + 0.3 * dc, r + 0.1 + 0.3 * dr),
                    xytext=(c - 0.3 * dc, r + 0.1 - 0.3 * dr),
                    arrowprops=dict(arrowstyle="-|>", lw=2,
                                    color=ink))
    ax.set_xticks(range(COLS),
                  [f"col {c}" for c in range(COLS)])
    ax.set_yticks(range(ROWS),
                  [f"row {r}" for r in range(ROWS)])
    ax.set_title(title)

print("setup ready -", device)
```

~~~text
setup ready - cpu
~~~

## Part A -- The Model Is Known: Value Iteration

### Step 1: The Transition Model

The environment is **stochastic**: an action gives a *distribution* over next squares. With
probability 0.8 the robot moves where it intended and 0.1 each to the two perpendicular directions;
a move into the wall or off the grid leaves it where it was. That distribution is the `P(s'|s,a)` of
the Markov Decision Process:

```python
def move(s, a):
    """Where action a leads from s, ignoring the noise."""
    r, c = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
    if 0 <= r < ROWS and 0 <= c < COLS and (r, c) != WALL:
        return (r, c)
    return s                    # bumped a wall or an edge

def transitions(s, a):
    """P(s'|s,a): 0.8 intended, 0.1 each sideways."""
    probs = {}
    for d, p in [(a, 0.8), (PERP[a][0], 0.1),
                 (PERP[a][1], 0.1)]:
        nxt = move(s, d)
        probs[nxt] = probs.get(nxt, 0.0) + p
    return probs

print("from (2,0) taking action N, the robot lands in:")
for nxt, p in transitions(START, "N").items():
    print(f"  {nxt} with probability {p}")
```

~~~text
from (2,0) taking action N, the robot lands in:
  (1, 0) with probability 0.8
  (2, 1) with probability 0.1
  (2, 0) with probability 0.1
~~~

From the bottom-left square, "go North" reaches the square above 80% of the time and slips East 10%.
The other 10% slips West, off the edge of the grid, so the robot stays where it is.

### Step 2: Value Iteration

The **value** of a square is the total reward the robot can expect from it when it plays its best.
The **Bellman optimality** equation says that a square's value is its own reward plus the discounted
value of the best action's outcomes, averaged over the noise:

`V*(s) = R(s) + gamma * max_a  sum_s' P(s'|s,a) V*(s')`

Each terminal is worth its own reward. **Value iteration** applies the update to every other square,
over and over, until the values stop changing. The function also tracks the **greedy policy** (the
best action under the current values) after every sweep, to see when it settles:

```python
def expected_next(V, s, a):
    """sum over s' of P(s'|s,a) * V(s')"""
    return sum(p * V[nxt]
               for nxt, p in transitions(s, a).items())

def greedy(V, s):
    return max(ACTIONS, key=lambda a: expected_next(V, s, a))

def value_iteration(living_reward, tol=1e-9):
    V = {s: TERMINALS.get(s, 0.0) for s in states}
    policy, settled = None, 0
    for sweep in range(1, 1001):
        new_V, delta = dict(V), 0.0
        for s in states:
            if s in TERMINALS:
                continue
            best = max(expected_next(V, s, a)
                       for a in ACTIONS)
            new_V[s] = living_reward + GAMMA * best
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        new_policy = {s: greedy(V, s) for s in states
                      if s not in TERMINALS}
        if new_policy != policy:
            policy, settled = new_policy, sweep
        if delta < tol:
            return V, policy, sweep, settled
    return V, policy, sweep, settled

V, pi_star, sweeps, settled = value_iteration(LIVING)
print(f"values converged after {sweeps} sweeps")
print(f"the policy stopped changing after {settled}\n")
for r in range(ROWS):
    print("  ".join(" WALL " if (r, c) == WALL
                    else f"{V[(r, c)]:+.3f}"
                    for c in range(COLS)))
```

~~~text
values converged after 37 sweeps
the policy stopped changing after 12

+0.812  +0.868  +0.918  +1.000
+0.762   WALL   +0.660  -1.000
+0.705  +0.655  +0.611  +0.388
~~~

These are the values of Russell & Norvig's classic example (their Figure 17.3), matched to three
decimals. Squares nearer the +1 goal are worth more. The square just left of the pit, (1, 2), is
worth noticeably less than its neighbours: one unlucky slip from there ends the episode at -1. Notice
also that the **policy** settled long before the **values** did: the best move in every square was
already right after a dozen sweeps, and the remaining sweeps only refined the numbers.

### Step 3: The Optimal Policy

The policy `pi*` is the greedy action over `V*`: from each square, take the action whose noisy
outcome has the highest expected value. Print it as arrows, then draw the values and the policy
together:

```python
ARROW = {"N": "^", "S": "v", "E": ">", "W": "<"}

def show_policy(policy):
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            if s == WALL:
                row.append("#")
            elif s in TERMINALS:
                row.append("+" if TERMINALS[s] > 0 else "-")
            else:
                row.append(ARROW[policy[s]])
        print(" ".join(row))

print("optimal policy (+ goal, - pit, # wall):")
show_policy(pi_star)

fig, ax = plt.subplots(figsize=(6.5, 4))
draw_grid(V, pi_star, "Value iteration: V* and pi*", ax)
plt.tight_layout()
plt.show()
```

~~~text
optimal policy (+ goal, - pit, # wall):
> > > +
^ # ^ -
^ < < <
~~~

Along the top row the robot drives East to the goal. The interesting square is (1, 2), just left of
the pit: its arrow points **North**, toward the top row, not East into the pit. North still carries a
10% chance of slipping East into the -1. The only risk-free move is West, bumping into the wall and
staying put (worth 0.641), but that wastes steps; at -0.04 a step, going North (0.660) is worth the
small risk. The bottom row takes the long way round, West and then up the left column, rather than
pass next to the pit.

### Step 4: From Values to Action-Values

`V*(s)` says how good a square is; the **action-value** `Q*(s, a)` says how good each *move* from
that square is: its reward plus the expected value of where the move lands. The two are linked by
`V*(s) = max_a Q*(s, a)`. Compute `Q*` at the start square:

```python
def q_value(V, s, a, living_reward=LIVING):
    return living_reward + GAMMA * expected_next(V, s, a)

q_star = {a: q_value(V, START, a) for a in ACTIONS}
for a, q in q_star.items():
    print(f"  Q*(start, {a}) = {q:+.3f}")
print(f"  V*(start)    = {V[START]:+.3f}  (best: "
      f"{max(q_star, key=q_star.get)})")
```

~~~text
  Q*(start, N) = +0.705
  Q*(start, S) = +0.660
  Q*(start, E) = +0.631
  Q*(start, W) = +0.671
  V*(start)    = +0.705  (best: N)
~~~

North is the best move (0.705), and the value of the square is exactly the value of that move. West
(0.671) and South (0.660) mostly bump the edge and leave the robot where it is, which costs a step.
East (0.631) is the worst: it moves the robot onto (2, 1), below the wall and off the left-column
route, and from there the best plan is to walk back West, so East wastes steps. `V` answers "how good is this square?"; `Q` answers "how good is each move?",
and `Q` is what an agent acts on. Part B learns exactly these four numbers without ever seeing
`P(s'|s,a)`.

## Part B -- The Model Is Unknown: Q-Learning

Value iteration needed the transition probabilities. A real agent usually does not have them: it can
only **try** a move and see where it lands. **Q-learning** works from such samples. After each move
`s, a -> s'` it nudges its estimate toward the **TD target**, the Bellman update computed from one
sample instead of the full distribution:

`Q(s,a) <- Q(s,a) + alpha * [ R(s) + gamma * max_a' Q(s',a') - Q(s,a) ]`

The bracket is the **TD error**. The step size `alpha` shrinks as a pair is visited more often, so the
estimate averages over the noisy outcomes. To choose moves, the agent uses **epsilon-greedy**: with
probability `epsilon` it tries a random action (explore), otherwise it takes the best one so far
(exploit). Epsilon starts at 1 and decays to 0.05 over the first half of training.

### Step 5: Learning Q from Samples

`sample_move` plays the environment. It uses the probabilities internally, but the agent sees only
the square it lands on:

```python
def sample_move(s, a, rng):
    """The environment: one random next square."""
    outcomes = transitions(s, a)
    return rng.choices(list(outcomes),
                       weights=list(outcomes.values()))[0]

def q_learning(episodes=20_000, epsilon=(1.0, 0.05),
               seed=0):
    rng = random.Random(seed)
    eps_start, eps_end = epsilon
    Q = {(s, a): 0.0 for s in states
         if s not in TERMINALS for a in ACTIONS}
    visits = collections.Counter()
    for ep in range(episodes):
        eps = max(eps_end,
                  eps_start * (1 - ep / (episodes / 2)))
        s = START
        while s not in TERMINALS:
            if rng.random() < eps:
                a = rng.choice(list(ACTIONS))     # explore
            else:                                 # exploit
                a = max(ACTIONS, key=lambda b: Q[(s, b)])
            nxt = sample_move(s, a, rng)
            if nxt in TERMINALS:
                future = TERMINALS[nxt]
            else:
                future = max(Q[(nxt, b)] for b in ACTIONS)
            td_target = LIVING + GAMMA * future
            visits[(s, a)] += 1
            alpha = 1 / visits[(s, a)] ** 0.6
            Q[(s, a)] += alpha * (td_target - Q[(s, a)])
            s = nxt
    return Q

Q = q_learning()
print("start square    learned Q   value iteration Q*")
for a in ACTIONS:
    print(f"  {a}             {Q[(START, a)]:+.3f}"
          f"      {q_star[a]:+.3f}")

pi_q = {s: max(ACTIONS, key=lambda b: Q[(s, b)])
        for s in states if s not in TERMINALS}
agree = sum(pi_q[s] == pi_star[s] for s in pi_q)
print(f"\npolicy agrees with value iteration in "
      f"{agree} of {len(pi_q)} squares")
```

~~~text
start square    learned Q   value iteration Q*
  N             +0.707      +0.705
  S             +0.661      +0.660
  E             +0.632      +0.631
  W             +0.671      +0.671

policy agrees with value iteration in 9 of 9 squares
~~~

Twenty thousand episodes of trial and error, and the learned action-values at the start square are
within 0.002 of the exact ones; the policy is identical in all 9 squares. Side by side, with the
learned `max_a Q(s, a)` as the colour:

```python
V_q = {s: (TERMINALS[s] if s in TERMINALS else
           max(Q[(s, a)] for a in ACTIONS)) for s in states}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
draw_grid(V, pi_star, "Value iteration (model known)",
          axes[0])
draw_grid(V_q, pi_q, "Q-learning (samples only)", axes[1])
plt.tight_layout()
plt.show()
```

The two grids show the same arrows. The values differ by at most about 0.02, most at (1, 2) beside
the pit (0.639 learned, 0.660 exact). One move in ten from there slips into the -1, so the sampled
targets of that square swing between about +0.9 and -1, the noisiest in the grid, and the shrinking
step size has not averaged that noise away completely. Q-learning never computed an expectation over
`P(s'|s,a)`; it averaged real outcomes instead.

## Part C -- Too Many States: A Deep Q-Network

The gridworld has 9 non-terminal squares, so `Q` fits in a table of 36 numbers (9 squares x 4
actions). A lander's state is **8 numbers**: six continuous ones (position, velocity, angle and
angular velocity) and two flags for whether each leg touches the ground. No table can hold every
combination. A **Deep Q-Network** replaces the table with a neural
network that takes the state and outputs one Q-value per action. The TD target from Part B becomes the
**label** of an ordinary regression, trained with the same loop as Unit V.

### Step 6: The Environment and a Random Agent

LunarLander has **4 actions**: do nothing, fire the left engine, fire the main engine, fire the right
engine. The reward is shaped: moving toward the pad and slowing down earn points, firing engines costs
a little, a crash costs -100, and landing at rest earns +100. An episode ends on a crash, on a landing,
or after 1000 steps; the task counts as **solved** when the average return reaches **200**.

As an MDP: the states S are the 8 numbers, the actions A are the 4 engine choices, the reward R comes
with each move, and gamma is 0.99. The transitions P are the physics simulator itself, which the
agent never sees: it can only act and observe, so the problem is **model-free**, like Part B.

First, a helper that plays one episode with any policy, and how a random agent does on 20 fresh
episodes:

```python
env = gym.make("LunarLander-v3")        # for training
test_env = gym.make("LunarLander-v3")   # for testing
N_STATE = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
print(f"state: {N_STATE} numbers | actions: {N_ACTIONS}")

def run_episode(policy, seed, environment=test_env):
    """Play one episode; return (return, steps, outcome)."""
    obs, _ = environment.reset(seed=seed)
    total, steps, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, _ = \
            environment.step(policy(obs))
        total, steps = total + reward, steps + 1
        done = terminated or truncated
    if truncated:
        outcome = "timeout"
    else:
        outcome = "landed" if reward > 0 else "crashed"
    return total, steps, outcome

UNSEEN = range(1000, 1020)   # 20 episodes never used before
coin = random.Random(0)
results = [run_episode(lambda obs: coin.randrange(N_ACTIONS),
                       seed) for seed in UNSEEN]
returns = [r for r, _, _ in results]
outcomes = collections.Counter(o for _, _, o in results)
print(f"random agent: average return {np.mean(returns):.1f}"
      f" | {dict(outcomes)}")
```

~~~text
state: 8 numbers | actions: 4
random agent: average return -214.9 | {'crashed': 20}
~~~

The random agent tumbles and crashes in all 20 episodes. That is the baseline the
network has to beat.

### Step 7: The Q-Network, the Replay Buffer and the Target Network

The Q-network is a small multilayer perceptron, 8 inputs to 4 outputs. Two more pieces make the
training stable enough to work:

- a **replay buffer** stores the last 50,000 moves `(s, a, r, s', done)`. Each update trains on a
  random batch drawn from it, so consecutive, nearly identical moves do not dominate a batch, and
  every move is reused many times;
- a **target network**, a frozen copy of the Q-network, computes the TD targets. It is refreshed every
  250 steps, so the label the network chases holds still between refreshes.

```python
def make_qnet():
    return nn.Sequential(
        nn.Linear(N_STATE, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, N_ACTIONS))

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
q_net = make_qnet().to(device)
target_net = make_qnet().to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = torch.optim.Adam(q_net.parameters(), lr=6.3e-4)
replay = collections.deque(maxlen=50_000)

n_params = sum(p.numel() for p in q_net.parameters())
print(f"Q-network: {N_STATE} -> 256 -> 256 -> {N_ACTIONS}"
      f" ({n_params:,} parameters)")
```

~~~text
Q-network: 8 -> 256 -> 256 -> 4 (69,124 parameters)
~~~

One training step is a regression. The prediction is `Q(s, a)` for the action actually taken; the
label is the TD target `r + gamma * max_a' Q_target(s', a')`, with the future dropped when the move
ended the episode. The loss is the Huber loss, which behaves like the squared error for small errors
and grows only linearly for large ones:

```python
GAMMA_RL = 0.99
BATCH = 128

def act(model, obs):
    """The greedy action: the largest predicted Q."""
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32,
                            device=device)
        return int(model(x).argmax().item())

def train_step():
    batch = random.sample(replay, BATCH)
    s, a, r, s2, done = (
        torch.as_tensor(np.array(col), dtype=torch.float32,
                        device=device)
        for col in zip(*batch))
    q_sa = q_net(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        best_next = target_net(s2).max(1).values
        td_target = r + GAMMA_RL * (1 - done) * best_next
    loss = F.smooth_l1_loss(q_sa, td_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10)
    optimizer.step()
```

<details><summary>(Optional) What the less familiar calls in train_step do</summary>

- `q_net(s)` returns a `(128, 4)` table: one row per move in the batch, one Q-value per action.
  `.gather(1, a.long().unsqueeze(1))` picks, in each row, the column of the action actually taken,
  and `.squeeze(1)` turns the result back into 128 numbers. Those are the predictions.
- `(1 - done)` is 0 for a move that ended the episode (a crash or a landing) and 1 otherwise, so the
  future term `GAMMA_RL * best_next` is dropped exactly when there is no future. A move cut off by the
  1000-step limit is stored with `done = 0`: the lander could have kept flying.
- `nn.utils.clip_grad_norm_(..., 10)` rescales the gradient whenever its overall size exceeds 10, so
  one batch with a huge TD error (early on, when the targets are wild) cannot throw the weights far
  off in a single step.
- `F.smooth_l1_loss` is the Huber loss: squared error for errors below 1, absolute error above.
</details>

### Step 8: Training, with a Safeguard

DQN training is **unstable**: the agent's test score can jump up and then collapse again a few
thousand steps later. So the loop tests the greedy agent on the same 10 episodes every 10,000 steps,
keeps a copy of the weights at every test (and of the best one so far), and stops only when **two
tests in a row** reach 200. A single good test is not enough: a network can get lucky on 10 episodes.
Epsilon decays from 1 to 0.05 over the first 36,000 steps, and the network takes one training step
every second move:

```python
WARMUP, UPDATE_EVERY, TARGET_EVERY = 1_000, 2, 250
EPS_END, EPS_DECAY = 0.05, 36_000
TEST_EVERY, GOAL, MAX_STEPS = 10_000, 200, 300_000
TEST_SEEDS = range(500, 510)

def evaluate(model, seeds):
    results = [run_episode(lambda obs: act(model, obs), s)
               for s in seeds]
    return (float(np.mean([r for r, _, _ in results])),
            [o for _, _, o in results])

obs, _ = env.reset(seed=SEED)
episode_return, curve, tests, snapshots = 0.0, [], [], {}
best_score, best_state, streak = -np.inf, None, 0
for step in range(1, MAX_STEPS + 1):
    eps = max(EPS_END, 1.0 - step * (1.0 - EPS_END) / EPS_DECAY)
    if random.random() < eps:
        action = random.randrange(N_ACTIONS)
    else:
        action = act(q_net, obs)
    nxt, reward, terminated, truncated, _ = env.step(action)
    replay.append((obs, action, reward, nxt,
                   float(terminated)))
    obs, episode_return = nxt, episode_return + reward
    if terminated or truncated:
        curve.append((step, episode_return))
        obs, _ = env.reset()
        episode_return = 0.0
    if step >= WARMUP and step % UPDATE_EVERY == 0:
        train_step()
    if step % TARGET_EVERY == 0:
        target_net.load_state_dict(q_net.state_dict())
    if step % TEST_EVERY == 0:
        score, _ = evaluate(q_net, TEST_SEEDS)
        tests.append((step, score))
        snapshots[step] = copy.deepcopy(q_net.state_dict())
        print(f"step {step:>7,} | epsilon {eps:.2f} | "
              f"test score {score:7.1f}")
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(q_net.state_dict())
        streak = streak + 1 if score >= GOAL else 0
        if streak == 2:
            print(f"stopped: two tests in a row >= {GOAL}")
            break
print(f"{len(curve)} training episodes; "
      f"best test score {best_score:.1f}")
```

~~~text
step  10,000 | epsilon 0.74 | test score   -51.2
step  20,000 | epsilon 0.47 | test score  -103.5
step  30,000 | epsilon 0.21 | test score   -30.5
step  40,000 | epsilon 0.05 | test score   -40.5
step  50,000 | epsilon 0.05 | test score   -40.0
step  60,000 | epsilon 0.05 | test score   -39.3
step  70,000 | epsilon 0.05 | test score   -15.1
step  80,000 | epsilon 0.05 | test score   239.0
step  90,000 | epsilon 0.05 | test score   266.3
stopped: two tests in a row >= 200
302 training episodes; best test score 266.3
~~~

For 70,000 steps the test score stays between -104 and -15, then it jumps to 239 and 266 and the
loop stops. This run is one of the quick ones. With other seeds, other thread counts or a GPU, the
score often reaches 200 and falls back, sometimes far below zero, before two tests in a row clear it.
One run of this exact code on a 12-thread CPU scored 230 at 70,000 steps, -28 at 80,000, 264 at
100,000 and -15 at 120,000, and needed 280,000 steps to stop. In runs made while preparing this
lab, stopping at the *first* test above 200 often kept a network that scored well below 200 on fresh
episodes; asking for two in a row helps, but it is not a guarantee, as the next step explains.

### Step 9: The Learning Curve

Every training episode's return, its running average, and the ten-episode tests:

```python
steps_at, ep_returns = zip(*curve)
running = np.convolve(ep_returns, np.ones(50) / 50,
                      mode="valid")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(steps_at, ep_returns, ".", ms=3, alpha=0.3,
        color="tab:blue", label="training episode return")
ax.plot(steps_at[49:], running, color="tab:blue",
        label="running average (50 episodes)")
test_steps, test_scores = zip(*tests)
ax.plot(test_steps, test_scores, "o-", color="tab:orange",
        label="greedy test (10 episodes)")
ax.axhline(GOAL, ls="--", color="0.3", label="goal: 200")
ax.set_xlabel("environment step")
ax.set_ylabel("return")
ax.set_title("DQN on LunarLander")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
```

The running average climbs from about -140 to about -40 within the first 25,000 steps and then sits
nearly flat for 45,000 more: the agent has learned **not to crash** long before it learns to land
(Exercise 3 shows what that stage looks like). The late jump in the tests runs ahead of the running
average, which still holds the earlier 50 episodes, and training episodes still take a random action
5% of the time.

### Step 10: An Honest Evaluation

The stopping rule chose its moment by looking at the 10 test episodes, so their score is optimistic,
just as a validation score is after it has been used to choose a model. The honest number comes from
episodes never used for any decision: the same 20 unseen episodes the random agent played.

```python
best_net = make_qnet().to(device)
best_net.load_state_dict(best_state)
results = [run_episode(lambda obs: act(best_net, obs), seed)
           for seed in UNSEEN]
returns = [r for r, _, _ in results]
outcomes = collections.Counter(o for _, _, o in results)
print(f"trained agent: average return {np.mean(returns):.1f}"
      f" | {dict(outcomes)}")
```

~~~text
trained agent: average return 265.7 | {'landed': 19, 'timeout': 1}
~~~

From -214.9 to 265.7 on the same 20 episodes: the task is solved. 19 episodes end in a landing; in the
20th the lander never crashes but runs out the 1000-step clock before it touches down. Here the
unseen score is almost the same as the best test score (266.3), but that is not guaranteed. The same
code run with two CPU threads instead of one passed two tests in a row (249, then 247) and then scored
161.0 on these 20 episodes, landing 15. Ten test episodes are a small sample, and a network chosen
because it did well on them is likely to have been a little lucky. That is why the honest number is the
one measured on episodes that played no part in choosing the network.

### Step 11: Watch It Fly

Numbers are one thing; watching is another. `episode_gif` plays policies side by side on the same
episode (the same terrain and the same starting push) and returns an animated GIF:

```python
render_env = gym.make("LunarLander-v3",
                      render_mode="rgb_array")

def episode_frames(policy, seed):
    obs, _ = render_env.reset(seed=seed)
    frames, total, done = [render_env.render()], 0.0, False
    while not done:
        obs, reward, terminated, truncated, _ = \
            render_env.step(policy(obs))
        frames.append(render_env.render())
        total += reward
        done = terminated or truncated
    if truncated:
        outcome = "timeout"
    else:
        outcome = "landed" if reward > 0 else "crashed"
    return frames, total, outcome

def episode_gif(policies, labels, seed):
    """Side-by-side animation; one panel per policy."""
    runs = [episode_frames(p, seed) for p in policies]
    longest = max(len(f) for f, _, _ in runs)
    stride = 2 if longest <= 500 else 4
    panels = []
    for (frames, total, outcome), label in zip(runs, labels):
        frames = frames[::stride]
        frames += [frames[-1]] * (
            (longest + stride - 1) // stride - len(frames))
        caption = f"{label}: {total:+.0f}, {outcome}"
        panels.append((frames, caption))
        print(caption)
    gif = []
    for i in range(len(panels[0][0])):
        canvas = Image.new("RGB", (300 * len(panels), 220),
                           "white")
        for k, (frames, caption) in enumerate(panels):
            img = Image.fromarray(frames[i]).resize((300, 200))
            canvas.paste(img, (300 * k, 20))
            ImageDraw.Draw(canvas).text(
                (300 * k + 6, 4), caption, fill="black")
        gif.append(canvas)
    buffer = io.BytesIO()
    gif[0].save(buffer, format="GIF", save_all=True,
                append_images=gif[1:], duration=40, loop=0)
    return buffer.getvalue()

SHOW = 1000                      # the first unseen episode
coin = random.Random(0)
gif = episode_gif(
    [lambda obs: coin.randrange(N_ACTIONS),
     lambda obs: act(best_net, obs)],
    ["random agent", "trained DQN"], SHOW)
display(GifImage(data=gif, format="gif"))
```

~~~text
random agent: -174, crashed
trained DQN: +292, landed
~~~

On the left, random thrusts tip the lander over and it crashes beside the pad. On the right, the
trained network fires the main engine in short bursts, corrects its tilt with the side engines, and
settles between the flags. Nobody told it how to fly: it only ever saw states, actions and rewards.

## Your Turn

### Exercise 1 -- How the Reward Shapes the Policy

The -0.04 living reward made the robot cautious: it detours around the pit. Make every step painful
(-2.0) and solve the gridworld again with `value_iteration`. Print the new policy with `show_policy`
and compare it with `pi_star`. **Before running it, predict:** which squares will change their
arrow, and why?

```python
# TODO: re-solve with a steep living reward and print the policy.
#   V_harsh, pi_harsh, sweeps_harsh, _ = value_iteration(-2.0)
#   print(f"converged after {sweeps_harsh} sweeps")
#   show_policy(pi_harsh)
```

<details><summary>Expected Output</summary>

~~~text
converged after 32 sweeps
> > > +
^ # > -
> > > ^
~~~

With every step costing -2.0, lingering is so expensive that the robot heads for the **nearest**
exit, even the -1 pit: ending the episode at -1 beats paying -2.0 for each extra step. The square
left of the pit now points **East, straight into it** (it pointed North, away from it, at -0.04), the
bottom-right square points up into it, and the bottom row runs East instead of taking the long way
round. Same environment, same dynamics; only the reward changed, and the optimal behaviour flipped
from cautious to reckless. **The reward is how you tell an agent what you want;** get it wrong and
the "optimal" policy does something you never intended.
</details>

### Exercise 2 -- Q-Learning Without Exploration

Q-learning found the optimal policy while exploring with epsilon-greedy. What happens if it never
explores? Train it again with `epsilon=(0.0, 0.0)`, so it always takes the action it currently thinks
is best, and count how many squares disagree with `pi_star`. **Before running it, predict:** will
the squares on the route from the start or the squares off it be the ones that go wrong?

```python
# TODO: train without exploration and compare the policies.
#   Q_greedy = q_learning(epsilon=(0.0, 0.0))
#   pi_greedy = {s: max(ACTIONS, key=lambda b: Q_greedy[(s, b)])
#                for s in pi_star}
#   wrong = [s for s in pi_star if pi_greedy[s] != pi_star[s]]
#   print("squares with a different action:", wrong)
#   show_policy(pi_greedy)
```

<details><summary>Expected Output</summary>

~~~text
squares with a different action: [(0, 2), (1, 2), (2, 1), (2, 2)]
> > ^ +
^ # < -
^ > ^ <
~~~

Four of the nine squares are wrong. The greedy learner found **a** route from the start (up the left
column, then East) and kept repeating it, so the moves it never tried kept their starting estimates
and it never learned they were better. Even on its own route it is wrong at (0, 2), one square from
the goal: it pushes North into the edge and relies on a 10% sideways slip to reach the +1, because
that was the first move that ever paid off. Squares off the route, such as (2, 1) and (2, 2), are
barely visited at all. With epsilon-greedy, the same 20,000 episodes found the optimal action in every
square: exploration is what lets the agent discover that an untried move is better than the one it
knows.
</details>

### Exercise 3 -- Find the Hover and the Setbacks

Training saved the network's weights at every test in `snapshots`. Score each snapshot on three
episodes and print what happened in each. Look for a snapshot whose episodes end in a **timeout**:
the lander has learned not to crash, but not yet to land. Then watch it next to the final agent with
`episode_gif`. **Then explain:** why would a reward with a -100 crash penalty make hovering the first
thing worth learning, and what does a snapshot whose score drops after rising tell you about when to
stop training?

```python
# TODO: score every snapshot, then watch one of them.
#   for step, state in snapshots.items():
#       net = make_qnet().to(device)
#       net.load_state_dict(state)
#       score, outcomes = evaluate(net, range(2024, 2027))
#       print(f"step {step:>6,}: {score:7.1f}  {outcomes}")
#
#   pick = ____   # a step whose episodes time out
#   early = make_qnet().to(device)
#   early.load_state_dict(snapshots[pick])
#   display(GifImage(data=episode_gif(
#       [lambda obs: act(early, obs),
#        lambda obs: act(best_net, obs)],
#       [f"step {pick:,}", "final"], 2024), format="gif"))
```

<details><summary>Expected Output</summary>

~~~text
step 10,000:   -48.0  ['timeout', 'timeout', 'timeout']
step 20,000:  -122.7  ['timeout', 'crashed', 'timeout']
step 30,000:   -17.6  ['timeout', 'timeout', 'timeout']
step 40,000:   -32.4  ['timeout', 'timeout', 'timeout']
step 50,000:     2.8  ['timeout', 'timeout', 'timeout']
step 60,000:    -6.8  ['timeout', 'timeout', 'timeout']
step 70,000:   -19.9  ['timeout', 'timeout', 'timeout']
step 80,000:   253.4  ['landed', 'landed', 'landed']
step 90,000:   277.6  ['landed', 'landed', 'landed']
~~~

With `pick = 40_000`, the animation prints:

~~~text
step 40,000: -27, timeout
final: +290, landed
~~~

From 10,000 to 70,000 steps almost every episode ends in a **timeout**: the network hovers above the
pad, firing its engines, for the full 1000 steps. Crashing costs -100 and the engines cost only a
little per step, so hovering is a safe, mediocre strategy, and the agent stays there for most of
training. At 20,000 steps it suffers a **setback**: one episode crashes, and that snapshot's 10-episode
test fell to -103.5. (This run never collapses after reaching the goal; the 12-thread run quoted in
Step 8 did, twice, which is exactly why one good test is not enough to stop.) Then, between 70,000 and 80,000 steps, it learns to put the lander down, and both later
snapshots land every time. Your snapshots will differ on a GPU or another machine, but a long hover
stage and at least one setback are typical.
</details>

## Summary

- A **Markov Decision Process** is the tuple (states, actions, transition probabilities `P(s'|s,a)`,
  rewards `R`, discount `gamma`). With the model known, **value iteration** applies the **Bellman
  optimality** update until the values settle: 37 sweeps here, matching Russell & Norvig to three
  decimals. The policy settled much earlier, after 12.
- `V*(s) = max_a Q*(s, a)`: the value of a square is the value of its best move, and `Q` is what an
  agent acts on.
- Without the model, **Q-learning** learns `Q` from sampled moves, nudging each estimate toward the
  **TD target** `R(s) + gamma * max_a' Q(s', a')`. It recovered value iteration's policy in every
  square, but only because **epsilon-greedy** kept it exploring (Exercise 2).
- When the state is continuous, a **Deep Q-Network** replaces the table: the TD target becomes a
  regression label, a **replay buffer** decorrelates the training batches, and a **target network**
  holds the label still. Training is **unstable**, so we kept the best weights and stopped only after
  two good tests in a row; the honest score comes from episodes never used for those decisions.
- **Reward shapes behaviour:** a -2.0 living reward sends the gridworld robot into the pit
  (Exercise 1), and the lander's reward makes "hover safely" an attractive stage on the way to "land"
  (Exercise 3).
