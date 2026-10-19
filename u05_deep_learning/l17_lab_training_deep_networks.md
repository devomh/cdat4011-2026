---
title: "Lab: The Deep-Network Training Toolkit"
unit: "V"
lesson: "17"
type: lab
tags: [deep-learning, pytorch, initialization, optimizers, batch-norm, dropout, vanishing-gradients]
difficulty: intermediate
duration: "60 mins"
---

**Goal:** train a network that would otherwise refuse to train. You will watch a deep net's signal
**vanish**, then bring it back to life with the right **initialization** and **activation**; race three
**optimizers**; and use **dropout** and **batch normalization** to fight overfitting and stabilize
training. Steps 1-4 run in seconds on CPU on small synthetic data, so you can see each *mechanism*
clearly. Step 5 then turns the whole toolkit on a real dataset -- **MNIST** handwritten digits -- and
puts the trained net up against three classic models from earlier units. (Convolutional networks
built for images are L18; full hyperparameter tuning is the CDAT 4013 lab.) Pairs with the concept
note [Training Deep Networks](l17_concept_training_deep_networks.qmd).

> **Previously:** L16 -- Backpropagation & Autodiff (we built the gradient engine)  |  **Next:** L18 -- Convolutional Neural Networks

> This page is the read-only view. To run the lab, open the notebook (`l17_lab_training_deep_networks.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u05_deep_learning/l17_lab_training_deep_networks.ipynb)

## Scenario

We can now build a network of any depth (L14-L15) and compute its gradients exactly (L16). But depth
is treacherous: stack enough layers and backprop's long product of local derivatives drives the
gradient to the lower layers toward zero, and the net never learns. This lab is the engineer's repair
kit -- initialization, activations, optimizers, learning rate, dropout, and batch normalization -- each
applied to a small net so you can see exactly what it fixes.

## Setup

Two cells: the first installs, the second imports, seeds, and defines the three helpers we reuse all
lab.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships numpy, scikit-learn, matplotlib, torch and torchvision, so this is effectively a no-op there.
%pip install -q numpy scikit-learn matplotlib torch torchvision
# local, in a terminal (not in the notebook):  uv add numpy scikit-learn matplotlib torch torchvision
```

```python
# Setup, cell 2 of 2 -- IMPORTS + SEEDS + HELPERS (safe to re-run without re-installing)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42); torch.manual_seed(42)

def make_split(X, y, seed=42):
    """Stratified 75/25 split, standardized (scaler fit on train only), returned as tensors."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    t = lambda a, d: torch.tensor(a, dtype=d)
    return t(Xtr, torch.float32), t(ytr, torch.long), t(Xte, torch.float32), t(yte, torch.long)

def deep_net(activation, in_dim=2, depth=6, width=30, he_init=False, dropout=0.0, batchnorm=False,
             out_dim=2):
    """A `depth`-hidden-layer MLP with `out_dim` classes; optional He init, BatchNorm, and Dropout."""
    layers, d = [], in_dim
    for _ in range(depth):
        linear = nn.Linear(d, width)
        if he_init:
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if batchnorm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

def train(model, data, optimizer, epochs=300, batch_size=None):
    """Full-batch training (batch_size=None) or shuffled mini-batches; returns the per-epoch
    loss curve (mean over the epoch's batches) and final train/test accuracy."""
    Xtr, ytr, Xte, yte = data
    loss_fn = nn.CrossEntropyLoss()
    curve = []
    for epoch in range(epochs):
        model.train()
        if batch_size is None:
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr), ytr)
            loss.backward()
            optimizer.step()
            curve.append(loss.item())
        else:
            running = 0.0
            for idx in torch.randperm(len(Xtr)).split(batch_size):
                optimizer.zero_grad()
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                loss.backward()
                optimizer.step()
                running += loss.item() * len(idx)
            curve.append(running / len(Xtr))
    model.eval()
    with torch.no_grad():
        tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
        te = (model(Xte).argmax(1) == yte).float().mean().item()
    return curve, tr, te

Xm, ym = make_moons(n_samples=1000, noise=0.20, random_state=42)
moons = make_split(Xm, ym, seed=42)
print(f"moons: {tuple(moons[0].shape)} train / {tuple(moons[2].shape)} test")
```

~~~text
moons: (750, 2) train / (250, 2) test
~~~

> **About exact numbers.** Every run is seeded, but PyTorch on a CPU splits its arithmetic across
> threads, and the order of those tiny rounding steps depends on the machine. Over hundreds of
> training steps the differences can grow, so your numbers -- on Colab especially -- may differ from
> the ones printed here, by a point or so of accuracy and more in a few places we flag. The
> conclusions do not change; where a comparison is too close to call, the text says so.

## Step 1: Watch the Signal Vanish

Before training anything, look at what depth does to a plain forward signal. Send a standard-normal
input through 20 `Linear -> ReLU` layers (no learning, just the forward pass) and measure the
activation standard deviation at the end under three weight initializations. If it collapses to 0, the
gradients flowing back will too.

```python
torch.manual_seed(0)
x = torch.randn(512, 100)
print(f"input std: {x.std().item():.4f}")
for name, init in [("PyTorch default", None),
                   ("Xavier / Glorot", nn.init.xavier_uniform_),
                   ("He / Kaiming",    nn.init.kaiming_uniform_)]:
    torch.manual_seed(0)
    h = x
    for _ in range(20):
        layer = nn.Linear(100, 100, bias=False)
        if init is not None:
            init(layer.weight)
        h = torch.relu(layer(h))
    print(f"{name:16s} -> std after 20 ReLU layers: {h.std().item():.4f}")
```

~~~text
input std: 0.9997
PyTorch default  -> std after 20 ReLU layers: 0.0000
Xavier / Glorot  -> std after 20 ReLU layers: 0.0009
He / Kaiming     -> std after 20 ReLU layers: 0.8975
~~~

The signal starting at std `1.0` is **gone** after 20 layers with the default or Xavier init -- and a
dead forward signal means dead backward gradients. **He / Kaiming** init, scaled for ReLU, keeps it
alive at `0.90`. Initialization is the first thing that decides whether a deep net can train at all.
(This demo uses the `_uniform_` initializers; the `deep_net` helper below uses `kaiming_normal_`.
Uniform vs normal changes only the sampling shape, not the variance scaling that keeps the signal
alive.)

## Step 2: Cure It -- Initialization + Activation

Now train a genuinely deep net (8 hidden layers) on the two-moons data, two ways: **sigmoid + the
default init** (the 2000s recipe) versus **ReLU + He init** (the modern one). We use plain SGD so the
choice of init and activation -- not an adaptive optimizer -- decides the outcome. First we print each
net's **first-layer gradient norm** (the smoking gun), then train and compare.

The `deep_net` helper already calls `nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")` when
`he_init=True` -- that one line is the entire fix.

```python
def first_layer_grad_norm(model, data):
    model.train()
    loss = nn.CrossEntropyLoss()(model(data[0]), data[1])
    loss.backward()
    return model[0].weight.grad.norm().item()

step2 = {}
for tag, activation, he in [("sigmoid + default", nn.Sigmoid, False),
                            ("ReLU + He",        nn.ReLU,    True)]:
    torch.manual_seed(1)
    net = deep_net(activation, depth=8, width=30, he_init=he)
    g = first_layer_grad_norm(net, moons)
    torch.manual_seed(1)
    net = deep_net(activation, depth=8, width=30, he_init=he)
    curve, tr, te = train(net, moons, torch.optim.SGD(net.parameters(), lr=0.5), epochs=400)
    step2[tag] = curve
    print(f"{tag:18s}: grad-norm {g:.6f}  loss {curve[0]:.3f} -> {curve[-1]:.3f}  train/test {tr:.3f}/{te:.3f}")
```

~~~text
sigmoid + default : grad-norm 0.000000  loss 0.693 -> 0.693  train/test 0.520/0.516
ReLU + He         : grad-norm 0.085034  loss 0.767 -> 0.066  train/test 0.977/0.988
~~~

The sigmoid net's first-layer gradient is `0.000000` -- the signal never reaches the bottom -- so its
loss is frozen at `0.693` (that is `ln 2`, the loss of pure guessing) and its accuracy stays near chance
(`0.52`). Trust the loss here, not the accuracy: the frozen net's outputs differ only by rounding
noise, so on another machine that one accuracy can land anywhere from about `0.5` to `0.65`, while the
loss stays at `0.693` every time. The ReLU + He net has a healthy gradient (`0.085`), its loss falls to `0.066`, and it reaches
`0.988` test accuracy. Same depth, same data, same optimizer -- the cure is initialization + activation.

```python
plt.figure(figsize=(6, 4))
for tag, curve in step2.items():
    plt.plot(curve, label=tag)
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend()
plt.title("A deep net won't train without the right init + activation")
plt.show()
```

## Step 3: Better Optimizers

With the net now trainable (ReLU + He), the **optimizer** decides how fast. Train the same 4-hidden-layer
net three ways -- plain SGD, SGD with momentum, and Adam -- from identical starting weights, and compare
the loss curves.

```python
step3 = {}
for tag, make_opt in [("SGD",          lambda p: torch.optim.SGD(p, lr=0.1)),
                      ("SGD+momentum", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
                      ("Adam",         lambda p: torch.optim.Adam(p, lr=0.01))]:
    torch.manual_seed(2)
    net = deep_net(nn.ReLU, depth=4, width=30, he_init=True)
    curve, tr, te = train(net, moons, make_opt(net.parameters()), epochs=150)
    step3[tag] = curve
    print(f"{tag:13s}: final loss {curve[-1]:.4f}  train/test {tr:.3f}/{te:.3f}")
```

~~~text
SGD          : final loss 0.1304  train/test 0.960/0.968
SGD+momentum : final loss 0.0558  train/test 0.976/0.988
Adam         : final loss 0.0440  train/test 0.984/0.980
~~~

Momentum (a velocity over past gradients) more than halves SGD's final loss; Adam (momentum + a
per-parameter adaptive step size) does best with no extra tuning. This is why Adam is the default
first choice.

```python
plt.figure(figsize=(6, 4))
for tag, curve in step3.items():
    plt.plot(curve, label=tag)
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend()
plt.title("Optimizers: momentum and Adam converge faster than plain SGD")
plt.show()
```

## Step 4: Dropout vs Overfitting -- completion problem

Switch to a harder, higher-dimensional set (20 features) where a wide net has the capacity to
**memorize**. Train it with no regularization first and watch it overfit -- train accuracy `1.000`,
test far behind.

```python
Xc, yc = make_classification(n_samples=300, n_features=20, n_informative=8, n_redundant=4,
                             n_clusters_per_class=2, class_sep=0.8, flip_y=0.05, random_state=0)
clf = make_split(Xc, yc, seed=0)

torch.manual_seed(3)
net = deep_net(nn.ReLU, in_dim=20, depth=3, width=256, he_init=True, dropout=0.0)
curve, tr, te = train(net, clf, torch.optim.Adam(net.parameters(), lr=0.001), epochs=600)
print(f"no dropout : train/test {tr:.3f}/{te:.3f}  gap {tr-te:.3f}")
```

~~~text
no dropout : train/test 1.000/0.840  gap 0.160
~~~

Now add **dropout** to regularize it. `deep_net` inserts an `nn.Dropout(p)` after each activation when
you pass `dropout=p`. Complete the dropout probability (try `0.5`) and compare:

```python
# Uncomment and complete the dropout probability, then compare to the run above:
# torch.manual_seed(3)
# net = deep_net(nn.ReLU, in_dim=20, depth=3, width=256, he_init=True, dropout=____)
# curve, tr, te = train(net, clf, torch.optim.Adam(net.parameters(), lr=0.001), epochs=600)
# print(f"with dropout: train/test {tr:.3f}/{te:.3f}  gap {tr-te:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
with dropout: train/test 1.000/0.880  gap 0.120
~~~

The blank is `0.5`. Dropout randomly zeros half the activations on every training step, so no unit can
rely on any other -- the net spreads its bets and generalizes better. Test accuracy rises from `0.840`
to `0.880` and the train-test gap shrinks from `0.160` to `0.120`. Note that `train()` activates
dropout and the final `model.eval()` (inside `train`) switches it off, so the test accuracy uses the
full network.
</details>

## Step 5: The Toolkit on Real Data

Moons and a synthetic 20-feature set show each mechanism cleanly, but does the toolkit matter on real
data? **MNIST** is the classic test: 70,000 handwritten digits (0-9), each a 28x28 grayscale image.
An MLP cannot see an image as a picture -- we **flatten** each one into 784 numbers (one per pixel,
scaled to [0, 1]) and treat it as an ordinary row of features. We train on a seeded 10,000-image
subset, so every model runs in seconds on CPU, and test on the full 10,000-image test set.

*Data provenance:* MNIST (LeCun, Cortes and Burges), downloaded at run time by `torchvision`, not
bundled with the course. The original MNIST page states no license; the dataset is commonly
redistributed under CC BY-SA 3.0. The first run downloads about 11 MB into `./data` and prints its progress;
later runs reuse the files.

```python
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

train_full = torchvision.datasets.MNIST("data", train=True, download=True)
test_full = torchvision.datasets.MNIST("data", train=False, download=True)

# a seeded 10,000-image training subset keeps every model fast on CPU
g = torch.Generator().manual_seed(0)
pick = torch.randperm(len(train_full), generator=g)[:10000]

# flatten each 28x28 image into 784 pixel features in [0, 1]
Xtr = train_full.data[pick].reshape(-1, 784).float() / 255
ytr = train_full.targets[pick]
Xte = test_full.data.reshape(-1, 784).float() / 255
yte = test_full.targets
mnist = (Xtr, ytr, Xte, yte)
print(f"train {tuple(Xtr.shape)} / test {tuple(Xte.shape)}")
print("train images per digit:", torch.bincount(ytr).tolist())
```

~~~text
train (10000, 784) / test (10000, 784)
train images per digit: [953, 1167, 1018, 1066, 967, 921, 920, 1045, 931, 1012]
~~~

Ten thousand rows of 784 pixel features, with roughly a thousand images of each digit -- balanced
enough that plain accuracy is a fair score.

Two helpers let you look at the data. `show_digits` draws random images with their labels;
`show_predictions` draws test images titled with the **true** label and the model's **predicted** one,
misses in red. Both take `n` (that many images in one row) or `grid=(rows, cols)` (a grid of
`rows * cols` images, which overrides `n`). `show_predictions` takes predictions, not a model, so it
works the same for a PyTorch net and a scikit-learn classifier; `which` picks `"correct"`, `"wrong"`,
or `"mixed"` (half of each, correct first).

```python
def _grid_axes(n, grid, cell_h):
    """A rows x cols grid of axes (grid=None: one row of n); returns (n, fig, flat axes)."""
    rows, cols = grid if grid is not None else (1, n)
    fig, axes = plt.subplots(rows, cols, figsize=(1.2 * cols, cell_h * rows), squeeze=False)
    return rows * cols, fig, axes.ravel()

def show_digits(X, y, n=10, grid=None, seed=0):
    """Plot random images from X (rows of 784 pixels) with their labels."""
    X, y = np.asarray(X), np.asarray(y)
    n, fig, axes = _grid_axes(n, grid, cell_h=1.6)
    pick = np.random.default_rng(seed).choice(len(X), n, replace=False)
    for ax, i in zip(axes, pick):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(str(y[i]))
        ax.axis("off")
    fig.tight_layout()
    plt.show()

def show_predictions(X, y_true, y_pred, n=10, grid=None, which="mixed", seed=0):
    """Plot test images titled 'true / pred', misses in red.
    which = "correct", "wrong", or "mixed" (half of each, correct first)."""
    X, y_true, y_pred = np.asarray(X), np.asarray(y_true), np.asarray(y_pred)
    n, fig, axes = _grid_axes(n, grid, cell_h=1.8)
    rng = np.random.default_rng(seed)
    right = np.flatnonzero(y_true == y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    if which == "correct":
        pick = rng.choice(right, n, replace=False)
    elif which == "wrong":
        pick = rng.choice(wrong, n, replace=False)
    else:
        pick = np.concatenate([rng.choice(right, n // 2, replace=False),
                               rng.choice(wrong, n - n // 2, replace=False)])
    for ax, i in zip(axes, pick):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        color = "black" if y_true[i] == y_pred[i] else "#d62728"
        ax.set_title(f"true {y_true[i]}\npred {y_pred[i]}", fontsize=9, color=color)
        ax.axis("off")
    fig.tight_layout()
    plt.show()

show_digits(Xtr, ytr, grid=(2, 5))
```

Every digit is easy for you to read, yet the same digit varies a lot in slant, stroke width and
shape -- compare the 3s and the 6s. The model gets none of that as a picture: each image arrives as
784 separate numbers.

### The cure ladder

Now train the **same** deep net -- 8 hidden layers of 128 units, 784 inputs, 10 outputs -- four times,
adding one cure each time: the 2000s recipe (sigmoid + default init + plain SGD), then ReLU + He
(Step 2), then Adam (Step 3), then BatchNorm + dropout (Cure 3 and Step 4). With 10,000 images we
now train in **mini-batches** of 128 (`batch_size=128`): each epoch is about 79 small steps instead
of one big one.
As in Step 2, we print each net's first-layer gradient norm before it trains.

```python
ladder = [
    ("sigmoid + default + SGD", nn.Sigmoid, False, False, 0.0, "SGD"),
    ("+ ReLU + He",             nn.ReLU,    True,  False, 0.0, "SGD"),
    ("+ Adam",                  nn.ReLU,    True,  False, 0.0, "Adam"),
    ("+ BatchNorm + Dropout",   nn.ReLU,    True,  True,  0.2, "Adam"),
]

def mnist_net(activation, he, bn, p):
    torch.manual_seed(5)
    return deep_net(activation, in_dim=784, depth=8, width=128, out_dim=10,
                    he_init=he, batchnorm=bn, dropout=p)

ladder_rows, nets = [], {}
for tag, activation, he, bn, p, opt_name in ladder:
    g = first_layer_grad_norm(mnist_net(activation, he, bn, p), mnist)
    net = mnist_net(activation, he, bn, p)
    if opt_name == "SGD":
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
    curve, tr, te = train(net, mnist, opt, epochs=10, batch_size=128)
    ladder_rows.append((tag, tr, te))
    nets[tag] = net
    print(f"{tag:23s}: grad-norm {g:.6f}  loss {curve[0]:.3f} -> {curve[-1]:.3f}  "
          f"train/test {tr:.3f}/{te:.3f}")
```

~~~text
sigmoid + default + SGD: grad-norm 0.000000  loss 2.307 -> 2.307  train/test 0.102/0.103
+ ReLU + He            : grad-norm 0.305358  loss 1.458 -> 0.224  train/test 0.921/0.892
+ Adam                 : grad-norm 0.305358  loss 0.786 -> 0.037  train/test 0.990/0.948
+ BatchNorm + Dropout  : grad-norm 0.294249  loss 1.986 -> 0.273  train/test 0.975/0.948
~~~

The disease is the same on real data. The 2000s recipe has a first-layer gradient of `0.000000`, its
loss sits at `2.307` -- about `ln 10 = 2.303`, the loss of guessing among ten digits -- and its test
accuracy is `0.103`, chance. **ReLU + He** brings the gradient back (`0.305`) and the net trains
(`0.892` here); **Adam** drives the training loss much lower (`0.224` -> `0.037`) and the test
accuracy up to `0.948`. (The `+ Adam` row shows the same starting gradient as the row above because it
is the same net at the start -- the optimizer only changes how it moves.)

These two middle rows are the ones that move most between machines: we have seen `+ ReLU + He`
anywhere from `0.89` to `0.94` and `+ Adam` from `0.935` to `0.955`. Adam's lower training loss holds
every time; its test lead over plain SGD can shrink to nothing. The first and last rows barely move.

The last rung is the honest one: **BatchNorm + dropout** lowers train accuracy from `0.990` to `0.975`
and leaves test accuracy where it was (`0.948`), so the train-test gap shrinks from `0.041` to
`0.027`. That is regularization doing its job -- less memorizing. Whether it also nudges test accuracy
up or down is within the run-to-run noise above, so do not read a gain or a loss into it. It does
make the result **steadier**: this row lands at `0.946`-`0.948` on every machine we tried. (Its
training loss is higher, `0.273`, because dropout switches units off while that loss is measured.)

### Three classic baselines

Is `0.948` good? Only a baseline can say. Fit three models from earlier units on the **same**
784-pixel rows: **logistic regression** (L04, a linear model), a **random forest** (L12) and an **RBF
SVM** (L08). Then print all seven models in one table.

```python
Xtr_np, ytr_np = Xtr.numpy(), ytr.numpy()
Xte_np, yte_np = Xte.numpy(), yte.numpy()
classic = {
    "logistic regression": LogisticRegression(max_iter=1000),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    "RBF SVM": SVC(),
}
classic_rows = []
for tag, model in classic.items():
    model.fit(Xtr_np, ytr_np)
    classic_rows.append((tag, model.score(Xtr_np, ytr_np), model.score(Xte_np, yte_np)))

for tag, tr, te in ladder_rows + classic_rows:
    print(f"{tag:23s}: train {tr:.3f}  test {te:.3f}  gap {tr - te:.3f}")
```

~~~text
sigmoid + default + SGD: train 0.102  test 0.103  gap -0.001
+ ReLU + He            : train 0.921  test 0.892  gap 0.029
+ Adam                 : train 0.990  test 0.948  gap 0.041
+ BatchNorm + Dropout  : train 0.975  test 0.948  gap 0.027
logistic regression    : train 0.966  test 0.909  gap 0.057
random forest          : train 1.000  test 0.952  gap 0.048
RBF SVM                : train 0.986  test 0.964  gap 0.022
~~~

Read down the test column. The toolkit took the deep net from chance (`0.103`) to about `0.95` -- and
it beats **logistic regression** (`0.909`), so depth buys something over a linear model. But the
**random forest** (`0.952`) is level with the best MLP rows (within about a point, either way,
depending on the run) and the **RBF SVM** (`0.964`) beats every one of them, with no init, no
optimizer, no learning rate to tune. They are cheap, too: the forest and the SVM each fit in seconds;
logistic regression is the slow one here (about 20 seconds on a laptop CPU).

Why does the deep net not win? **Flattening threw the image away.** The MLP sees 784 separate
columns: nothing in it knows that pixel 300 sits next to pixel 301, and a digit shifted two pixels to
the right lands on entirely different columns. On those terms, with 10,000 examples, a forest or an
SVM does just as well. The toolkit makes a deep net **trainable**; it does not make it **better** than
a strong classic model. What changes that is an architecture that keeps the 2-D layout -- the
convolutional network of L18.

### Look at the mistakes

Accuracy is one number; the images show *what kind* of errors a model makes. First the full-toolkit
MLP -- the top row is digits it got right, the bottom row digits it missed:

```python
with torch.no_grad():
    mlp_pred = nets["+ BatchNorm + Dropout"](Xte).argmax(1)
rf_pred = classic["random forest"].predict(Xte_np)
print(f"MLP misses {int((mlp_pred != yte).sum())} of {len(yte)} test digits; "
      f"forest misses {int((rf_pred != yte_np).sum())}")
show_predictions(Xte, yte, mlp_pred, grid=(2, 5))
```

~~~text
MLP misses 522 of 10000 test digits; forest misses 484
~~~

Now only the forest's misses:

```python
show_predictions(Xte, yte, rf_pred, grid=(2, 5), which="wrong")
```

Some misses are digits a person might pause over too: in the MLP's bottom row, a 9 with an open loop
read as a 4. Others are plain blunders: two 7s written with a crossbar, called a 3 and a 2. In the
forest's grid, the 8s stand out -- clear 8s called 9, 7 or 3 -- next to a cleanly written 2 called
a 3. A model that scores each pixel column on its own sees *how much ink falls where*, not the two
closed loops that make an 8 an 8. Change the `seed` (or `which="correct"`) to look at more.

## Your Turn

### Exercise 1 -- The learning-rate sweep

The learning rate is the single most important hyperparameter. On the moons net (4 hidden, ReLU + He),
train with Adam at `lr` in `[10, 1, 0.01, 0.0001]` for 150 epochs and read which diverges, which
crawls, and which wins. Print the final loss and train/test accuracy for each.

**Hint:** loop over the learning rates; re-seed (`torch.manual_seed(4)`) and rebuild the net each time
so only the learning rate changes; reuse `train(net, moons, torch.optim.Adam(net.parameters(), lr=lr), epochs=150)`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
lr=10.0   : final loss 4.7964  train/test 0.736/0.716
lr=1.0    : final loss 0.4204  train/test 0.799/0.780
lr=0.01   : final loss 0.0480  train/test 0.981/0.992
lr=0.0001 : final loss 0.4205  train/test 0.845/0.900
~~~

The solution loops over the four rates, re-seeding and rebuilding the net each time so only `lr`
changes: `for lr in [10.0, 1.0, 0.01, 0.0001]:` then `torch.manual_seed(4)`,
`net = deep_net(nn.ReLU, depth=4, width=30, he_init=True)`, and
`train(net, moons, torch.optim.Adam(net.parameters(), lr=lr), epochs=150)`. `lr=10` is so large the
loss diverges (ends at `4.80`, accuracy near chance); `lr=0.0001` is so small it barely moves in 150
epochs (`0.42`); `lr=0.01` is in the sweet spot (test `0.992`). When a net will not train, sweep the
learning rate by orders of magnitude first. (The two unstable rates, `lr=10` and `lr=1`, are the rows
that change most between machines -- an unstable run amplifies every rounding difference. The
`lr=0.01` and `lr=0.0001` rows come out the same everywhere.)
</details>

### Exercise 2 -- Add batch normalization

Insert batch normalization into the deep (8-hidden) net and compare it to the plain version. `deep_net`
adds an `nn.BatchNorm1d` after each linear layer when you pass `batchnorm=True`. Train both with Adam
(`lr=0.01`) for 200 epochs and compare the final loss and test accuracy.

**Hint:** loop over `batchnorm in [False, True]`; re-seed (`torch.manual_seed(5)`) and rebuild with
`deep_net(nn.ReLU, depth=8, width=30, he_init=True, batchnorm=batchnorm)` each time.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
plain         : final loss 0.0963  train/test 0.967/0.968
with BatchNorm: final loss 0.0003  train/test 1.000/0.984
~~~

The solution loops `for tag, bn in [("plain        ", False), ("with BatchNorm", True)]:`, then
re-seeds with `torch.manual_seed(5)`, rebuilds via
`deep_net(nn.ReLU, depth=8, width=30, he_init=True, batchnorm=bn)`, and trains with
`torch.optim.Adam(net.parameters(), lr=0.01)` for 200 epochs.
Batch normalization keeps each layer's activations well-conditioned, so the deep net drives its loss
far lower (`0.0003` vs `0.0963`) and fits a touch better. (This exercise is one of the places the
numbers move most between machines -- the plain net's loss ranges from about `0.06` to `0.24` -- but
the BatchNorm net's loss is far lower every time.) Like dropout, BatchNorm behaves differently
in `train()` (batch statistics) and `eval()` (running statistics) -- the `train` helper already handles
both modes for you.
</details>

### Exercise 3 -- Depth vs width on MNIST

Step 5's MLP was 8 layers deep. **Predict first:** would a *shallow* net -- 2 hidden layers of 256
units -- do better or worse on MNIST than the 8-layer one, and would it catch the RBF SVM? Write your
guess down, then train it with the full Step 5 recipe (ReLU + He, Adam `lr=0.001`, 10 epochs,
`batch_size=128`) and print its train/test accuracy. Then, in one or two sentences, explain the result
using what flattening does to an image.

**Hint:** re-seed with `torch.manual_seed(5)`, then
`deep_net(nn.ReLU, in_dim=784, depth=2, width=256, out_dim=10, he_init=True)`, and reuse
`train(net, mnist, torch.optim.Adam(net.parameters(), lr=0.001), epochs=10, batch_size=128)`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
2 x 256: train/test 0.999/0.962
~~~

The solution re-seeds with `torch.manual_seed(5)`, builds
`net = deep_net(nn.ReLU, in_dim=784, depth=2, width=256, out_dim=10, he_init=True)`, trains with
`curve, tr, te = train(net, mnist, torch.optim.Adam(net.parameters(), lr=0.001), epochs=10, batch_size=128)`
and prints `f"2 x 256: train/test {tr:.3f}/{te:.3f}"` (expect about `0.96` on your machine). The
**shallow** net wins: test `0.962` against the 8-layer net's `0.948`, level with the RBF SVM
(`0.964`), and it trains faster. On
flattened pixels depth was never the advantage -- the 8-layer net needed the whole toolkit just to
train, and still did worse. Depth pays when each layer can build on structure the one below found
(edges, then strokes, then digits), which is what convolutional layers are designed for (L18).
</details>

## Summary

- **Unstable gradients** are the core obstacle to depth: backprop's long product of local derivatives
  **vanishes** (factors < 1) or **explodes** (factors > 1). A deep sigmoid net's first-layer gradient
  was `0.000000` and its loss froze at `0.693` (chance).
- **Initialization + activation** are the first cure: He / Kaiming init with ReLU keeps the signal
  alive (std `0.90` vs `0.00` after 20 layers) and lets the same deep net reach `0.988`.
- **Optimizers** decide speed: momentum then Adam cut the final loss well below plain SGD
  (`0.044` vs `0.130`) from the same start.
- The **learning rate** is the master dial -- too high diverges (`lr=10` -> loss `4.80`), too low
  crawls (`lr=0.0001`), `lr=0.01` wins (`0.992`).
- **Dropout** regularizes a memorizing net (test `0.840` -> `0.880`, gap `0.160` -> `0.120`) and
  **batch normalization** stabilizes a deep one (loss `0.096` -> `0.0003`); both differ in
  `train()` vs `eval()` mode, so always `model.eval()` before measuring.
- **On real data (MNIST, flattened pixels)** the toolkit took an 8-layer MLP from chance (`0.103`) to
  about `0.95`, yet a random forest (`0.952`), an RBF SVM (`0.964`) and a 2-layer MLP (`0.962`) did as
  well or better. The toolkit makes depth *trainable*; an architecture that keeps the image's layout
  (L18) is what makes it pay.
- Next (L18): a new architecture built for a specific data type -- **convolutional** networks for images.
