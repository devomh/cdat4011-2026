---
title: "Lab: Training a Multilayer Perceptron in PyTorch"
unit: "V"
lesson: "15"
type: lab
tags: [neural-networks, mlp, pytorch, universal-approximation, training-loop]
difficulty: intermediate
duration: "60 mins"
---

**Goal:** stop hand-wiring weights and start *learning* them. You will build a multilayer
perceptron in **PyTorch**, count and draw its parameters, train it with the explicit four-step
loop (forward, loss, backward, step), and watch it learn a curved decision boundary that a linear
model cannot. Then you will look inside: what the hidden layer does to the data so that the
output layer can separate it. Finally you will see **universal approximation** in action: a
one-hidden-layer network fits a wiggly curve better as you give it more units.
`loss.backward()` does the calculus for us here -- *how* it computes those gradients is the next
lesson (L16). Pairs with the concept note
[Multilayer Perceptrons](l15_concept_multilayer_perceptron.qmd).

> **Previously:** L14 -- From Perceptrons to Neural Networks (we hand-wired an XOR net)  |  **Next:** L16 -- Backpropagation & Autodiff (opens the `backward()` black box)

> This page is the read-only view. To run the lab, open the notebook (`l15_lab_multilayer_perceptron.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u05_deep_learning/l15_lab_multilayer_perceptron.ipynb)

## Scenario

In L14 we solved XOR by setting every weight by hand. Here we hand the job to gradient descent.
Our classification dataset is **concentric circles** (`make_circles`): one class forms a ring
around the other, so -- exactly like XOR -- **no straight line separates them**. A linear model
will score chance; an MLP with one hidden layer will learn the ring. For the
universal-approximation demo we fit a **wiggly 1-D curve** (`y = sin(3 pi x)`) and watch the fit
sharpen as the hidden layer grows.

Everything is seeded, small, and CPU-only, so it trains in seconds and your numbers will match
this page.

## Setup

Two cells: the first only installs, the second imports, seeds, and builds the data.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships torch, scikit-learn and matplotlib, so this is effectively a no-op there.
%pip install -q torch scikit-learn matplotlib
# local, in a terminal (not in the notebook):  uv add torch scikit-learn matplotlib
```

```python
# Setup, cell 2 of 2 -- IMPORTS, SEEDS, DATA (safe to re-run without re-installing)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

# Concentric circles: one class ringed by the other -- NOT linearly separable (a 2-D XOR).
X, y = make_circles(n_samples=400, noise=0.10, factor=0.5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)                      # fit on TRAIN only (no leakage)
X_tr_t = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
X_te_t = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
y_tr_t = torch.tensor(y_tr)
y_te_t = torch.tensor(y_te)
print(len(X_tr), "train /", len(X_te), "test;  features:", X.shape[1], " classes:", len(np.unique(y)))
```

~~~text
300 train / 100 test;  features: 2  classes: 2
~~~

## Step 1: Build an MLP in PyTorch

A model in PyTorch is a stack of layers. `nn.Sequential` chains them: a `nn.Linear(2, 16)` dense
layer (2 inputs to 16 hidden units), a `nn.ReLU()` activation, then a `nn.Linear(16, 2)` output
layer producing **2 logits** (one score per class). Build it and look at its structure:

```python
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 16),    # input (2 features) -> hidden layer of 16 units
    nn.ReLU(),           # nonlinear activation (the ingredient L14 showed is essential)
    nn.Linear(16, 2),    # hidden -> output: 2 logits (one per class)
)
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters()))

with torch.no_grad():                       # an untrained forward pass: 2 logits per point
    print("output shape for 3 points:", tuple(model(X_te_t[:3]).shape), "(2 logits per point)")
```

~~~text
Sequential(
  (0): Linear(in_features=2, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=2, bias=True)
)
trainable parameters: 82
output shape for 3 points: (3, 2) (2 logits per point)
~~~

Where does 82 come from? List every parameter tensor with its shape:

```python
for name, p in model.named_parameters():
    shape, n = str(tuple(p.shape)), p.numel()
    print(f"{name:9s} shape {shape:8s} {n:3d} parameters")
```

~~~text
0.weight  shape (16, 2)   32 parameters
0.bias    shape (16,)     16 parameters
2.weight  shape (2, 16)   32 parameters
2.bias    shape (2,)       2 parameters
~~~

The prefix is the layer's position in the `Sequential`: `0` is the first `nn.Linear`, `2` the
second. The `ReLU` at position 1 has no parameters -- it is a fixed function. Notice the weight's
shape is **[out, in]**: `0.weight` is 16 x 2, one row of 2 input weights for each hidden unit.
That gives the counting rule for any dense layer `nn.Linear(n_in, n_out)`:

> **parameters = n_in * n_out (weights) + n_out (biases) = (n_in + 1) * n_out**

Each of the `n_out` units has one weight per input plus its own bias. Add the layers up:
`(2 + 1) * 16 = 48` and `(16 + 1) * 2 = 34`, total **82**. Right now they are random, so the
logits are meaningless -- training is what makes them useful.

The same network as a picture: the helper below draws one circle per unit and one line per
weight, and writes each layer's count above its gap. Run it; reading its code is optional -- it
only reads the layer sizes off the model you give it.

```python
# Helper: draw an nn.Sequential MLP as a diagram.
def draw_mlp(model):
    """One circle per unit, one gray line per weight."""
    linears = [m for m in model if isinstance(m, nn.Linear)]
    sizes = [linears[0].in_features]
    sizes += [layer.out_features for layer in linears]
    acts = [type(m).__name__ for m in model
            if not isinstance(m, nn.Linear)]
    fig, ax = plt.subplots(figsize=(2.6 * len(sizes), 5))
    ys = [np.linspace(-1, 1, n) if n > 1 else np.zeros(1)
          for n in sizes]
    # the weights: every unit to every unit of the next layer
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        for y0 in ys[i]:
            for y1 in ys[i + 1]:
                ax.plot([i, i + 1], [y0, y1], color="gray",
                        lw=0.5, alpha=0.5, zorder=1)
        n_par = (n_in + 1) * n_out
        count = f"{n_in}x{n_out} + {n_out}\n= {n_par}"
        ax.text(i + 0.5, 1.18, count, ha="center", fontsize=9)
    # the units, with a label under each layer
    for i, n in enumerate(sizes):
        ax.scatter([i] * n, ys[i], s=180, color="white",
                   edgecolors="k", zorder=2)
        if i == 0:
            label = f"input\n{n} features"
        elif i == len(sizes) - 1:
            label = f"output\n{n} logits"
        else:
            unit = "unit" if n == 1 else "units"
            label = f"hidden\n{n} {unit} ({acts[i - 1]})"
        ax.text(i, -1.35, label, ha="center", va="top",
                fontsize=9)
    total = sum(p.numel() for p in model.parameters())
    arch = " -> ".join(str(n) for n in sizes)
    ax.set_title(f"{arch} MLP: {total} trainable parameters",
                 pad=34)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-1.7, 1.5)
    ax.axis("off")
    plt.show()

draw_mlp(model)
```

Each gray line is one weight: the 32 on the left are `0.weight`, the 32 on the right `2.weight`.
The biases are not drawn -- there is one inside every hidden and output circle (16 + 2 = 18).

**Predict, then check.** A deeper network with *two* hidden layers of 8 units: 2 -> 8 -> 8 -> 2.
Count its parameters with the rule **before** you run anything, then let PyTorch check you:

```python
# Uncomment, replace ____ with YOUR count by hand, and run.
# my_count = ____
# deeper = nn.Sequential(
#     nn.Linear(2, 8), nn.ReLU(),
#     nn.Linear(8, 8), nn.ReLU(),
#     nn.Linear(8, 2),
# )
# n = sum(p.numel() for p in deeper.parameters())
# print("my count:", my_count, "  PyTorch's count:", n)
# draw_mlp(deeper)
```

<details><summary>Expected Output</summary>

~~~text
my count: 114   PyTorch's count: 114
~~~

Layer by layer: `(2 + 1) * 8 = 24`, `(8 + 1) * 8 = 72`, `(8 + 1) * 2 = 18`, total **114**. The
8 -> 8 layer between the two hidden layers holds most of them: a dense layer's count grows with
the *product* of the sizes it connects. The diagram shows the same three counts above its gaps.
</details>

## Step 2: Train It -- the Four-Step Loop

Training is one loop repeated for many **epochs** (passes over the data). Each step is four lines:
**forward** (predict), **loss** (measure error), **backward** (compute gradients), **step** (nudge
the weights downhill). `nn.CrossEntropyLoss` takes the **raw logits** -- it applies the softmax
internally, so we must *not* add one ourselves.

```python
loss_fn = nn.CrossEntropyLoss()                       # takes raw logits; applies log-softmax itself
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1, 401):
    optimizer.zero_grad()           # gradients accumulate -- clear last step's first
    logits = model(X_tr_t)          # 1. forward pass
    loss = loss_fn(logits, y_tr_t)  # 2. measure the error
    loss.backward()                 # 3. backward: autograd computes every gradient (HOW -> L16)
    optimizer.step()                # 4. gradient-descent step (L03), down the slopes
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

with torch.no_grad():
    acc_tr = (model(X_tr_t).argmax(1) == y_tr_t).float().mean().item()
    acc_te = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
print(f"train accuracy {acc_tr:.3f}  test accuracy {acc_te:.3f}")
```

~~~text
epoch 100  loss 0.0127
epoch 200  loss 0.0057
epoch 300  loss 0.0031
epoch 400  loss 0.0020
train accuracy 1.000  test accuracy 1.000
~~~

The loss falls and the network reaches a perfect **1.000** on the held-out test set. The single
new line that does the heavy lifting is `loss.backward()`: it computes how to change all 82
parameters at once. We treat it as a black box for now -- L16 opens it up (it is the
backpropagation algorithm). Plot the boundary it learned -- a closed ring, not a straight line.
The plotting code goes in a small helper, `plot_boundary`, because we will reuse it: it asks the
model for a class at every point of a fine grid and shades the grid by the answer.

```python
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 300),
                     np.linspace(-2.5, 2.5, 300))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                    dtype=torch.float32)

def plot_boundary(model, ax, title=""):
    """Shade each grid point by the class the model predicts,
    then overlay the test points."""
    with torch.no_grad():
        Z = model(grid).argmax(1).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                alpha=0.3, cmap="coolwarm")
    ax.scatter(X_te_t[:, 0], X_te_t[:, 1], c=y_te,
               cmap="coolwarm", edgecolors="k", s=25)
    ax.set_xlabel("x1 (standardized)")
    ax.set_ylabel("x2 (standardized)")
    ax.set_title(title)

fig, ax = plt.subplots(figsize=(5, 5))
plot_boundary(model, ax,
              "The MLP learned a closed (ring) boundary")
plt.show()
```

The red region is where the model predicts the inner class, and it encloses every inner (red)
test point while leaving the outer ring (blue) outside: a closed boundary, which no single straight
line can draw.

### Watch it learn

That plot shows only the end of training. To see the boundary *form*, rebuild the same network
from the same seed -- an identical starting point -- and photograph its boundary at a few moments
during the same 400 epochs. This is exactly Step 2's training run, just photographed along the way.

```python
torch.manual_seed(0)    # same start as Step 1's model
watch = nn.Sequential(nn.Linear(2, 16), nn.ReLU(),
                      nn.Linear(16, 2))
opt = torch.optim.Adam(watch.parameters(), lr=0.05)
snapshots = [0, 5, 10, 20, 50, 400]

fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
panels = iter(axes.ravel())
# epoch = how many updates the net has had so far
for epoch in range(401):
    if epoch in snapshots:
        with torch.no_grad():
            loss_now = loss_fn(watch(X_tr_t), y_tr_t).item()
            pred = watch(X_te_t).argmax(1)
            acc = (pred == y_te_t).float().mean().item()
        print(f"after {epoch:3d} epochs  train loss"
              f" {loss_now:.4f}  test accuracy {acc:.3f}")
        title = f"after {epoch} epochs (test acc {acc:.2f})"
        plot_boundary(watch, next(panels), title)
    if epoch < 400:
        opt.zero_grad()
        loss_fn(watch(X_tr_t), y_tr_t).backward()
        opt.step()
plt.tight_layout()
plt.show()
```

~~~text
after   0 epochs  train loss 0.7185  test accuracy 0.500
after   5 epochs  train loss 0.5356  test accuracy 0.590
after  10 epochs  train loss 0.3853  test accuracy 0.970
after  20 epochs  train loss 0.1525  test accuracy 1.000
after  50 epochs  train loss 0.0275  test accuracy 1.000
after 400 epochs  train loss 0.0020  test accuracy 1.000
~~~

Untrained (0 epochs), the boundary is an arbitrary cut and the accuracy is a coin flip (0.500).
By epoch 5 a small blob has appeared in the middle, by epoch 10 it is a rough ring (0.970), and by
epoch **20** the test set is perfect. After that the picture barely moves -- yet the training loss
keeps falling, from 0.1525 at epoch 20 to 0.0020 at epoch 400. The test accuracy is already
perfect, so those later epochs mostly push the probability of the true class toward 1 (log loss,
L04, rewards confidence) and tidy up the last few training points. That is also why the snapshots
are bunched at the start: photographing every 100 epochs
would show four identical rings.

## Step 3: The Hidden Layer Is the Difference -- completion problem

Was it the hidden layer that did it, or just "more PyTorch"? Train a **linear** model -- one
`nn.Linear` straight from inputs to logits, with **no hidden layer and no activation** -- on the
same circles. Complete the output size (2 logits) and run it:

```python
# Uncomment and complete: a LINEAR model has NO hidden layer and NO activation.
# torch.manual_seed(0)
# linear_model = nn.Sequential(nn.Linear(2, ____))    # 2 inputs straight to the logits -- no hidden layer
# opt = torch.optim.Adam(linear_model.parameters(), lr=0.05)
# for epoch in range(400):
#     opt.zero_grad()
#     loss_fn(linear_model(X_tr_t), y_tr_t).backward()
#     opt.step()
# with torch.no_grad():
#     acc = (linear_model(X_te_t).argmax(1) == y_te_t).float().mean().item()
# print(f"linear model (no hidden layer) test accuracy {acc:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
linear model (no hidden layer) test accuracy 0.550
~~~

The blank is `nn.Linear(2, 2)`. With no hidden layer the model is a single linear classifier, and
it scores **0.550** -- barely above the coin flip -- because a straight line cannot wrap around a
ring. This is exactly L14's lesson, now *learned* rather than hand-wired: the nonlinear hidden
layer is what makes the difference, and gradient descent found the weights for us.
</details>

## Step 4: What the Hidden Layer Does

Step 3 showed that the hidden layer is what wins. But what does it actually *do* to the data? With
16 hidden units we cannot look: the hidden layer turns each point into 16 numbers, a point in a
16-dimensional space. So shrink the hidden layer to **3** units -- still enough to wrap the ring --
and each point's hidden values `h = (h1, h2, h3)` become a point we *can* plot, in 3-D.

`small[0]` and `small[1]` are the first two layers of the `Sequential` (the `nn.Linear` and the
`ReLU`), so `small[1](small[0](x))` runs the forward pass only as far as the hidden layer:

```python
torch.manual_seed(0)
small = nn.Sequential(nn.Linear(2, 3), nn.ReLU(),
                      nn.Linear(3, 2))   # only 3 hidden units
opt = torch.optim.Adam(small.parameters(), lr=0.05)
for epoch in range(400):
    opt.zero_grad()
    loss_fn(small(X_tr_t), y_tr_t).backward()
    opt.step()

with torch.no_grad():
    # hidden layer only: h = ReLU(W1 x + b1), a row per point
    H_te = small[1](small[0](X_te_t)).numpy()
    pred = small(X_te_t).argmax(1)
acc = (pred == y_te_t).float().mean().item()
print(f"3 hidden units: test accuracy {acc:.3f}")
for c, name in [(1, "inner circle"), (0, "outer ring")]:
    mean_h = H_te[y_te == c].mean(axis=0)
    mean_h = tuple(round(v, 2) for v in mean_h.tolist())
    print(f"{name:12s} (class {c}): average h = {mean_h}")
```

~~~text
3 hidden units: test accuracy 0.990
inner circle (class 1): average h = (0.58, 0.69, 1.04)
outer ring   (class 0): average h = (1.33, 1.52, 1.74)
~~~

Three units still score 0.990. Inner-circle points get *smaller* hidden values than outer-ring
points: about 0.6 to 1.0 per unit on average, against 1.3 to 1.7. Now look at both spaces side by
side, the original input space on the left and the hidden space on the right, to see why.

```python
W1 = small[0].weight.detach().numpy()
b1 = small[0].bias.detach().numpy()
W2 = small[2].weight.detach().numpy()
b2 = small[2].bias.detach().numpy()

fig = plt.figure(figsize=(13, 5.5))
# Left: input space, plus the line where each unit switches on
ax1 = fig.add_subplot(1, 2, 1)
plot_boundary(small, ax1,
              "Input space: each hidden unit draws a line")
xs = np.array([-2.5, 2.5])
for k, style in enumerate(["-", "--", ":"]):
    (w1, w2), b = W1[k], b1[k]
    # the line w.x + b = 0: unit k is 0 on one side of it
    ax1.plot(xs, -(w1 * xs + b) / w2, style, color="k",
             lw=1.8, label=f"h{k + 1} switches on")
ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5)
ax1.legend(loc="lower left")

# Right: hidden space, plus the output layer's decision plane
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(H_te[:, 0], H_te[:, 1], H_te[:, 2], c=y_te,
            cmap="coolwarm", edgecolors="k", s=25)
# class 1 wins where logit1 - logit0 = w.h + c > 0 (a plane)
w, c = W2[1] - W2[0], b2[1] - b2[0]
g1, g2 = np.meshgrid(np.linspace(0, H_te[:, 0].max(), 30),
                     np.linspace(0, H_te[:, 1].max(), 30))
g3 = -(w[0] * g1 + w[1] * g2 + c) / w[2]
g3[(g3 < 0) | (g3 > H_te[:, 2].max())] = np.nan
ax2.plot_surface(g1, g2, g3, alpha=0.25, color="gray")
ax2.set_xlabel("h1"); ax2.set_ylabel("h2")
ax2.set_zlabel("h3")
ax2.set_title("Hidden space: a flat plane splits the classes")
ax2.view_init(elev=20, azim=225)
plt.show()
```

**Left -- the input space.** A ReLU unit outputs 0 on one side of its line `w.x + b = 0` and grows
on the other side. The three black lines are where `h1`, `h2` and `h3` switch on, and the learned
boundary is a polygon that bends only where it crosses one of them: each hidden unit contributes
one fold line. The boundary crosses each line twice, so three lines give a six-cornered polygon --
enough to close a shape around the inner circle.

Now look where the three lines cross: almost at the centre of the inner circle, like spokes. So
every point switches on one or two units, and the farther it is from the centre, the larger their
values grow. Inner points stay small; ring points grow large.

**Right -- the hidden space.** The same test points, moved to their hidden coordinates. The inner
circle lands in the corner near the origin, and the outer ring is pushed out along the axes, far
over the "on" side of at least one line. The points lying flat on the walls and axes are ReLU at
work -- a unit that is off outputs exactly 0. And now a **flat plane** (the gray triangle) cuts the
inner corner off from everything else: in effect, it asks "are your hidden values small?", which
here means "are you near the centre?" (To see it from another side,
change `azim` in `view_init` and re-run.)

That plane is the output layer. It is the same kind of model that failed in Step 3 -- a linear
classifier -- but it is applied to `h` instead of `x`. The hidden layer does not classify anything
itself; it moves the points to new coordinates in which one straight cut is enough. You met a
version of this idea in L08: the kernel trick also lifts the data into more dimensions, where a
flat boundary separates it. The difference is who chooses the lift -- for the SVM *we* pick the
kernel, while the MLP **learns** its mapping from the data, with the same gradient descent that
fits everything else.

## Step 5: Universal Approximation

The reason MLPs are worth the trouble: a single hidden layer, given enough units, can approximate
**any** continuous function. Watch it happen -- fit `y = sin(3 pi x)` with a one-hidden-layer net
at three widths and read the final MSE:

```python
xg = torch.linspace(-1, 1, 200).unsqueeze(1)
yg = torch.sin(3 * np.pi * xg)

fits = {}
for H in (1, 4, 16):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, H), nn.Tanh(), nn.Linear(H, 1))   # 1 -> H hidden -> 1
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(3000):
        opt.zero_grad()
        mse = nn.functional.mse_loss(net(xg), yg)
        mse.backward()
        opt.step()
    fits[H] = net
    print(f"hidden units H={H:2d}: final MSE {mse.item():.4f}")
```

~~~text
hidden units H= 1: final MSE 0.4649
hidden units H= 4: final MSE 0.0153
hidden units H=16: final MSE 0.0002
~~~

```python
plt.figure(figsize=(7, 4))
plt.plot(xg, yg, "k--", linewidth=2, label="true: sin(3 pi x)")
for H, net in fits.items():
    with torch.no_grad():
        plt.plot(xg, net(xg), label=f"H={H}")
plt.legend(); plt.xlabel("x"); plt.ylabel("y")
plt.title("Wider hidden layer = closer fit (universal approximation)")
plt.show()
```

One hidden unit (`H=1`) can barely bend -- MSE 0.465. Four units already track the wave (0.0153),
and sixteen nail it (0.0002). More width buys a closer fit: that is universal approximation made
concrete. (It is an *existence* result -- it promises a fit exists, not that it is easy to find or
that one wide layer is the most efficient shape; depth usually is.)

## Your Turn

### Exercise 1 -- The capacity dial

Re-run the **circles** MLP from Steps 1-2 but shrink the hidden layer to a single unit
(`nn.Linear(2, 1)` then `nn.ReLU()` then `nn.Linear(1, 2)`), and compare its test accuracy to the
16-unit version. What does too little capacity do? Then *look* at the small network: draw it with
`draw_mlp` and plot its boundary with `plot_boundary`. Using Step 4, explain why one hidden unit
cannot close a ring.

**Hint:** copy Step 1-2, change only the two hidden sizes to 1; keep `torch.manual_seed(0)`, Adam lr 0.05, 400 epochs. To look (calling the small model `tiny`): `draw_mlp(tiny)`, then `fig, ax = plt.subplots(figsize=(5, 5))`, `plot_boundary(tiny, ax, "H=1")`, `plt.show()`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
hidden units H= 1: test accuracy 0.500
hidden units H=16: test accuracy 1.000
~~~

A single hidden unit cannot bend enough to wrap the ring, so it **underfits** to chance (0.500,
just like the linear model), while 16 units reach a perfect 1.000. Capacity is a dial: too few
units underfit; too many risk overfitting (L09). Sixteen is plenty here.

The pictures show why. `draw_mlp(tiny)` reports a 2 -> 1 -> 2 net with just 7 parameters, and
the boundary plot shows no ring at all: one hidden unit means one line (Step 4), so the network
can make only a single straight cut. It places that cut in a corner and calls nearly every point
the inner class. One fold cannot close a ring.
</details>

### Exercise 2 -- Swap the activation

Rebuild the 16-unit circles MLP with `nn.Tanh()` in place of `nn.ReLU()` (same training), and
report its test accuracy. Does the network still solve the circles?

**Hint:** copy Step 1-2, change `nn.ReLU()` to `nn.Tanh()`; keep `torch.manual_seed(0)`, Adam lr 0.05, 400 epochs.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
Tanh hidden layer: test accuracy 1.000
~~~

Tanh reaches a perfect 1.000 too. The point is the *architecture* -- a hidden layer with **some**
nonlinear activation -- not one specific function. ReLU is the modern default (L14) for speed and
deep-network behavior, but the depth-plus-nonlinearity recipe is what carries the idea.
</details>

## Summary

- An **MLP** is a stack of dense layers with nonlinear activations: `nn.Sequential(nn.Linear,
  nn.ReLU, nn.Linear)`. Ours had 82 trainable parameters.
- **Counting parameters:** a dense layer `nn.Linear(n_in, n_out)` has `(n_in + 1) * n_out`
  (weights stored as [out, in], plus one bias per unit); sum over the layers. 2 -> 16 -> 2 is
  48 + 34 = 82; 2 -> 8 -> 8 -> 2 is 114.
- The **four-step training loop** -- `zero_grad` -> forward -> loss -> `backward()` -> `step` --
  *learned* the weights; `loss.backward()` is the autograd black box (its mechanism is L16).
  The boundary formed within about 20 epochs; later epochs mostly added confidence.
- A hidden layer let the MLP learn a **ring boundary** (test 1.000) that a linear model could not
  (0.550) -- L14's nonlinearity lesson, now learned by gradient descent instead of hand-wired.
- **What the hidden layer does:** each ReLU unit adds one fold line in the input space; in the
  hidden space (3 units, test 0.990) the classes become separable by a flat plane -- the output
  layer is a linear classifier working on `h` instead of `x`.
- **Universal approximation:** a wider hidden layer fit the wiggly curve better (MSE 0.465 ->
  0.0153 -> 0.0002), and too few units underfit (Ex1: H=1 scores chance). Tanh worked as well as
  ReLU (Ex2) -- the architecture is the point.
- Next (L16): open the `backward()` black box -- the backpropagation algorithm that computes all
  those gradients.
