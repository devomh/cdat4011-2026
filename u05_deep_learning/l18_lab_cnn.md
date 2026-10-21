---
title: "Lab: Build a CNN for Image Classification"
unit: "V"
lesson: "18"
type: lab
tags: [deep-learning, pytorch, cnn, convolution, pooling, image-classification]
difficulty: intermediate
duration: "60 mins"
---

**Goal:** see a convolution do its work, then build a small **convolutional neural network** and watch
it beat a plain MLP on images. You will convolve and pool by hand and confirm PyTorch agrees, watch a
filter act as an edge detector, assemble a `Conv -> ReLU -> Pool` network, trace its shapes and count
its weights, train it on Fashion-MNIST, and finally look inside it at what its filters respond to.
Pairs with the concept note [Convolutional Neural Networks](l18_concept_cnn.qmd).

> **Previously:** L17 -- Training Deep Networks (init, optimizers, regularizers)  |  **Next:** L19 -- CNN Architectures & Transfer Learning

> This page is the read-only view. To run the lab, open the notebook (`l18_lab_cnn.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u05_deep_learning/l18_lab_cnn.ipynb)

> **Run this notebook on a GPU.** In Colab: *Runtime > Change runtime type > T4 GPU*, then run
> the cells in order. Every training run here takes seconds on a GPU; on a CPU the whole lab still
> works, it just takes several minutes. The outputs shown on this page were produced on a CPU. On a
> GPU, or on another machine, your accuracies will differ slightly (typically in the third decimal),
> because the order in which floating-point operations are added up changes the result a little and
> training amplifies it. The comparisons and conclusions are the same.

## Scenario

An MLP flattens a 28x28 image into 784 numbers and forgets which pixel sat next to which. A CNN keeps
the grid: it slides small learnable **filters** across the image, building **feature maps** that say
where each pattern fires, and **pools** to shrink them. We will first strip convolution and pooling
down to arithmetic on a 4x4 patch, then scale up to a real classifier on Fashion-MNIST, measure the
gap against the MLP, check that the gap is not just a matter of size, and open the trained network up.

## Setup

Two cells: the first installs, the second imports, picks the device, loads the data and defines the
training helpers.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships torch, torchvision and matplotlib, so this is effectively a no-op there.
%pip install -q torch torchvision matplotlib
# local, in a terminal (not in the notebook):  uv add torch torchvision matplotlib
```

```python
# Setup, cell 2 of 2 -- IMPORTS + DEVICE + DATA + HELPERS (safe to re-run)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# use the GPU when the runtime has one
device = "cuda" if torch.cuda.is_available() else "cpu"

train_raw = torchvision.datasets.FashionMNIST(
    "data", train=True, download=True)
test_raw = torchvision.datasets.FashionMNIST(
    "data", train=False, download=True)
CLASSES = train_raw.classes

def to_tensors(ds):
    """uint8 images -> floats in [0, 1], shape [N, 1, 28, 28]."""
    X = ds.data.float().div(255).unsqueeze(1)
    return X, ds.targets

X_all, y_all = to_tensors(train_raw)
X_test, y_test = to_tensors(test_raw)

# 90/10 split of the 60,000 training images
g = torch.Generator().manual_seed(42)
perm = torch.randperm(60000, generator=g)
X_train, y_train = X_all[perm[:54000]], y_all[perm[:54000]]
X_val, y_val = X_all[perm[54000:]], y_all[perm[54000:]]

# move everything to the device once
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

def predict(model, X):
    """Predicted class for every image in X (2000 at a time)."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 2000]).argmax(1)
                          for i in range(0, len(X), 2000)])

def accuracy(model, X, y):
    return (predict(model, X) == y).float().mean().item()

def n_params(model):
    return sum(p.numel() for p in model.parameters())

def fit(make_model, epochs, seed=0, lr=1e-3, verbose=True):
    """Build a fresh model under a fixed seed, train it with Adam
    on mini-batches of 64, and record loss and train/val
    accuracy after every epoch."""
    torch.manual_seed(seed)
    model = make_model().to(device)
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        hist["loss"].append(total / len(X_train))
        hist["train_acc"].append(accuracy(model, X_train, y_train))
        hist["val_acc"].append(accuracy(model, X_val, y_val))
        if verbose:
            print(f"  epoch {epoch:2d}  "
                  f"loss {hist['loss'][-1]:.3f}  "
                  f"train {hist['train_acc'][-1]:.3f}  "
                  f"val {hist['val_acc'][-1]:.3f}")
    return model, hist

print("device:", device)
print(f"train {len(X_train)} / validation {len(X_val)} / "
      f"test {len(X_test)} images; {len(CLASSES)} classes")
```

~~~text
device: cpu
train 54000 / validation 6000 / test 10000 images; 10 classes
~~~

A few things in that cell matter later:

- **The data is loaded once as tensors.** `ds.data` holds all 60,000 training images as a
  `uint8` array of 0-255 pixel values; dividing by 255 scales them to [0, 1] (the feature scaling of
  L08), and `unsqueeze(1)` adds the **channel** dimension, giving the `[N, 1, 28, 28]` layout a
  convolution expects (N images, 1 channel because they are grayscale, 28x28 pixels). Moving the
  tensors to the device once, instead of converting every image as it is used, keeps the GPU busy.
- **Three splits.** The 60,000 training images are cut 90/10 into **54,000 to train on** and
  **6,000 for validation** (L09's validation set: we watch it while training). The separate
  **10,000-image test set** is used once per model, at the end.
- **`fit` builds the model itself, under a fixed seed.** It takes a function that makes a model
  (like `make_cnn` below), seeds the weights and the batch order, trains, and returns the trained
  model plus a history of loss and train/validation accuracy per epoch. Because each call seeds
  itself, re-running any training cell on its own gives the same result as running the notebook top
  to bottom.

The data is **Fashion-MNIST** (Zalando Research, MIT licence), downloaded by `torchvision` the
first time (about 30 MB).

## Step 1: Look at the Data

Two helpers, the same ones you met in L17, let you look at images. `show_images` draws random
images with their class names; `show_predictions` draws images titled with the **true** class and
the model's **predicted** one, misses in red. Both take `n` (that many images in one row) or
`grid=(rows, cols)` (overrides `n`). `show_predictions` takes predictions, not a model; `which` picks
`"correct"`, `"wrong"`, or `"mixed"` (half of each, correct first).

```python
def _grid_axes(n, grid, cell_h):
    """A rows x cols grid of axes (grid=None: one row of n)."""
    rows, cols = grid if grid is not None else (1, n)
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(1.4 * cols, cell_h * rows))
    return rows * cols, fig, axes.ravel()

def show_images(X, y, n=10, grid=None, seed=0):
    """Plot random images from X with their class names."""
    X, y = X.cpu().numpy(), y.cpu().numpy()
    n, fig, axes = _grid_axes(n, grid, cell_h=1.6)
    pick = np.random.default_rng(seed).choice(len(X), n, replace=False)
    for ax, i in zip(axes, pick):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(CLASSES[y[i]], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    plt.show()

def show_predictions(X, y_true, y_pred, n=10, grid=None,
                     which="mixed", seed=0):
    """Plot images titled 'true / pred', misses in red.
    which = "correct", "wrong", or "mixed" (half of each)."""
    X = X.cpu().numpy()
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    n, fig, axes = _grid_axes(n, grid, cell_h=1.9)
    rng = np.random.default_rng(seed)
    right = np.flatnonzero(y_true == y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    if which == "correct":
        pick = rng.choice(right, n, replace=False)
    elif which == "wrong":
        pick = rng.choice(wrong, n, replace=False)
    else:
        pick = np.concatenate([
            rng.choice(right, n // 2, replace=False),
            rng.choice(wrong, n - n // 2, replace=False)])
    for ax, i in zip(axes, pick):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"true {CLASSES[y_true[i]]}\n"
                     f"pred {CLASSES[y_pred[i]]}", fontsize=7,
                     color="black" if ok else "#d62728")
        ax.axis("off")
    fig.tight_layout()
    plt.show()

show_images(X_train, y_train, grid=(2, 5))
```

Ten classes of clothing and footwear, each a small grayscale picture. Some classes are easy to tell
apart (a trouser and a sandal share almost nothing), but four of them are all **upper-body garments**
-- T-shirt/top, pullover, coat and shirt -- with similar outlines at this resolution. Keep those four
in mind; they will come back when we look at the CNN's mistakes.

## Step 2: Convolution by Hand

A convolution is multiply-and-sum: lay a small **kernel** over each patch of the image, multiply
element-wise, and sum (the concept note writes this as a formula). Zero-pad the border so the output
keeps the input's size, and step with a **stride** of 1. Here is a 4x4 image `M` convolved with a
kernel whose middle row is ones:

```python
def convolve2D(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    # zero padding keeps the output the same size as the input
    padded = np.pad(image, ((ph, ph), (pw, pw)), "constant")
    out = np.zeros_like(image, dtype=np.float32)
    for i in range(ih):
        for j in range(iw):
            # multiply-and-sum one patch (stride 1)
            patch = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(patch * kernel)
    return out

M = np.array([[10, 24, 20, 175], [8, 5, 4, 184],
              [184, 174, 223, 201], [55, 89, 55, 192]],
             dtype=np.float32)
K = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
print("feature map M * K =\n", convolve2D(M, K))

Mt = torch.tensor(M).reshape(1, 1, 4, 4)
Kt = torch.tensor(K).reshape(1, 1, 3, 3)
same = np.allclose(F.conv2d(Mt, Kt, padding=1)[0, 0].numpy(),
                   convolve2D(M, K))
print("F.conv2d agrees:", same)
```

~~~text
feature map M * K =
 [[ 34.  54. 219. 195.]
 [ 13.  17. 193. 188.]
 [358. 581. 598. 424.]
 [144. 199. 336. 247.]]
F.conv2d agrees: True
~~~

Only the kernel's middle row is non-zero, so each output is the sum of a pixel and its **left and
right neighbors**: the top-left value is 0 + 10 + 24 = 34 (the 0 is padding), and the one beside it
is 10 + 24 + 20 = 54. PyTorch's `F.conv2d` does exactly the same thing on 4-D `NCHW` tensors (batch,
channels, height, width), just fast.

**Output size.** Padding and stride decide how big the feature map is. For an `n x n` input, a
`k x k` kernel, padding `p` and stride `s`, each side of the output is
`floor((n + 2p - k) / s) + 1`. Before running the next cell, predict the output size of each of the
three configurations below for our 4x4 image and 3x3 kernel.

```python
for pad, stride in [(1, 1), (0, 1), (1, 2)]:
    out = F.conv2d(Mt, Kt, padding=pad, stride=stride)
    print(f"padding={pad}, stride={stride}: "
          f"output {tuple(out.shape[2:])}")
print("stride 2 keeps every other value:\n",
      F.conv2d(Mt, Kt, padding=1, stride=2)[0, 0].numpy())
```

~~~text
padding=1, stride=1: output (4, 4)
padding=0, stride=1: output (2, 2)
padding=1, stride=2: output (2, 2)
stride 2 keeps every other value:
 [[ 34. 219.]
 [358. 598.]]
~~~

With padding 1 and stride 1, (4 + 2 - 3) / 1 + 1 = 4: the size is kept. Without padding,
(4 - 3) / 1 + 1 = 2: a 3x3 kernel fits in only two positions per row. With stride 2,
floor(3 / 2) + 1 = 2: the kernel jumps two pixels at a time, so the output keeps every other value
of the stride-1 map (34, 219, 358, 598). A strided convolution is one way to shrink a feature map;
pooling, in Step 4, is the other.

## Step 3: A Filter Is a Pattern Detector

Filters detect patterns. Send a synthetic bright square through a **vertical-edge** kernel (negative
column, zero, positive column) and read where it fires:

```python
square = torch.zeros(1, 1, 6, 6)
square[0, 0, 1:5, 1:5] = 1.0           # a white square on black
v_kernel = torch.tensor([[[[-1., 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]]]])  # vertical-edge detector
fmap = F.conv2d(square, v_kernel, padding=1)
print("feature map shape:", tuple(fmap.shape))
print("vertical-edge map:\n", fmap[0, 0].numpy())
```

~~~text
feature map shape: (1, 1, 6, 6)
vertical-edge map:
 [[ 1.  1.  0.  0. -1. -1.]
 [ 2.  2.  0.  0. -2. -2.]
 [ 3.  3.  0.  0. -3. -3.]
 [ 3.  3.  0.  0. -3. -3.]
 [ 2.  2.  0.  0. -2. -2.]
 [ 1.  1.  0.  0. -1. -1.]]
~~~

The filter is **positive on the square's left edge, negative on its right edge, and zero across the
flat interior** -- it responds to vertical edges and stays silent on flat regions. Because the same
weights are applied at every position, it would find the same edge anywhere: move the square and
the response moves with it (the concept note calls this **equivariance**). In a real CNN these
filters are not hand-set; they are **learned by backprop**.

```python
plt.figure(figsize=(7, 3))
plt.subplot(1, 2, 1)
plt.imshow(square[0, 0], cmap="gray")
plt.title("input square")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(fmap[0, 0], cmap="gray")
plt.title("vertical-edge feature map")
plt.axis("off")
plt.show()
```

In the feature map, white is a strong positive response (the left edge), black a strong negative one
(the right edge), and the mid-gray band in the middle is zero: the flat interior gives the filter
nothing to respond to.

## Step 4: Pooling by Hand

A **pooling** layer shrinks each feature map by summarizing small windows. **Max pooling** keeps the
largest value in each 2x2 window; **average pooling** keeps the mean. Apply both to the Step 2
feature map:

```python
fm = torch.tensor(convolve2D(M, K)).reshape(1, 1, 4, 4)
print("max pool 2x2:\n", F.max_pool2d(fm, 2)[0, 0].numpy())
print("average pool 2x2:\n", F.avg_pool2d(fm, 2)[0, 0].numpy())
print("weights in nn.MaxPool2d(2):", n_params(nn.MaxPool2d(2)))
```

~~~text
max pool 2x2:
 [[ 54. 219.]
 [581. 598.]]
average pool 2x2:
 [[ 29.5  198.75]
 [320.5  401.25]]
weights in nn.MaxPool2d(2): 0
~~~

Check the top-left window by hand: it holds 34, 54, 13 and 17, so max pooling keeps **54** and
average pooling gives (34 + 54 + 13 + 17) / 4 = **29.5**. The 4x4 map becomes 2x2: height and width
halve, which cuts the work of every later layer by four. And pooling has **no weights** -- there is
nothing to learn, it only summarizes.

## Step 5: Build a CNN, Trace Its Shapes, Count Its Weights

Stack two `Conv -> ReLU -> MaxPool` blocks, then `Flatten` and two dense layers. Padding keeps each
conv's height and width; only pooling shrinks them, so the size halves at each pool (28 -> 14 -> 7).
Trace the shapes to find the flattened dimension that feeds the first dense layer:

```python
def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),                        # 28 -> 14
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),                        # 14 -> 7
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),  # 64*7*7 = 3136
        nn.Linear(128, 10),                     # logits
    )

cnn = make_cnn()
x = torch.zeros(1, 1, 28, 28)
for layer in cnn:
    x = layer(x)
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d,
                          nn.Flatten, nn.Linear)):
        name = layer.__class__.__name__
        print(f"  after {name:9s}: {tuple(x.shape)}")
print("CNN total params:", n_params(cnn))
```

~~~text
  after Conv2d   : (1, 32, 28, 28)
  after MaxPool2d: (1, 32, 14, 14)
  after Conv2d   : (1, 64, 14, 14)
  after MaxPool2d: (1, 64, 7, 7)
  after Flatten  : (1, 3136)
  after Linear   : (1, 128)
  after Linear   : (1, 10)
CNN total params: 421642
~~~

**Counting a convolution's weights.** A filter spans every input channel, so one `k x k` filter on
`C_in` channels has `C_in * k * k` weights plus one bias, and the layer has `C_out` such filters:
`(C_in * k * k + 1) * C_out` parameters. Image size does not appear in the formula at all -- that is
parameter sharing.

```python
for i in (0, 3):                     # the two Conv2d layers
    conv = cnn[i]
    c_in, c_out = conv.in_channels, conv.out_channels
    k = conv.kernel_size[0]
    by_hand = (c_in * k * k + 1) * c_out
    print(f"Conv2d({c_in}, {c_out}, {k}): ({c_in}*{k}*{k} + 1)"
          f" * {c_out} = {by_hand}; PyTorch: {n_params(conv)}")
```

~~~text
Conv2d(1, 32, 3): (1*3*3 + 1) * 32 = 320; PyTorch: 320
Conv2d(32, 64, 3): (32*3*3 + 1) * 64 = 18496; PyTorch: 18496
~~~

The two convolutions hold only **320** and **18,496** weights. A dense layer wiring the 784 pixels to
just 128 units would already need 100,480.

Before training, make sure you can read the remaining numbers straight off the shapes: the
flattened size that feeds the first dense layer, and that layer's weight count (the L15 rule,
`(n_in + 1) * n_out`):

```python
# Your turn (warm-up): fill both blanks, then uncomment and run.
# flat_dim = ____               # 64 feature maps of 7 x 7
# head = ____                   # weights of nn.Linear(flat_dim, 128)
# assert flat_dim == cnn[7].in_features   # cnn[7] is the first nn.Linear
# assert head == n_params(cnn[7])
# print("flattened dimension:", flat_dim, " dense-layer weights:", head)
```

The blanks are `64 * 7 * 7 = 3136` and `(3136 + 1) * 128 = 401,536`. That one dense layer holds
401,536 of the CNN's 421,642 parameters (95%): in this network the convolutions are cheap and the
dense head is where the weights are.

## Step 6: Train It, and Beat the MLP

Train the CNN for 5 epochs. `fit` prints the loss and the train/validation accuracy after each epoch;
the test set is scored once at the end.

```python
cnn, cnn_hist = fit(make_cnn, epochs=5)
print(f"CNN test accuracy: {accuracy(cnn, X_test, y_test):.4f}")
```

~~~text
  epoch  1  loss 0.463  train 0.885  val 0.881
  epoch  2  loss 0.297  train 0.907  val 0.898
  epoch  3  loss 0.252  train 0.912  val 0.900
  epoch  4  loss 0.218  train 0.922  val 0.905
  epoch  5  loss 0.194  train 0.940  val 0.918
CNN test accuracy: 0.9130
~~~

Now the L15-style **MLP** on exactly the same images, with the same training settings:

```python
def make_mlp():
    return nn.Sequential(
        nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

mlp, mlp_hist = fit(make_mlp, epochs=5)
print(f"MLP params: {n_params(mlp)}")
print(f"MLP test accuracy: {accuracy(mlp, X_test, y_test):.4f}")
```

~~~text
  epoch  1  loss 0.536  train 0.861  val 0.852
  epoch  2  loss 0.373  train 0.881  val 0.869
  epoch  3  loss 0.335  train 0.870  val 0.852
  epoch  4  loss 0.309  train 0.893  val 0.875
  epoch  5  loss 0.290  train 0.901  val 0.882
MLP params: 235146
MLP test accuracy: 0.8753
~~~

The CNN is clearly ahead. But it also has **more** parameters than this MLP (421,642 against
235,146), so a fair objection is that it simply has more weights to fit with. Test that directly:
give an MLP as many weights as the CNN, and give a CNN fewer weights than the MLP.

```python
def make_mlp_wide():
    # one wide hidden layer: about as many weights as the CNN
    return nn.Sequential(
        nn.Flatten(), nn.Linear(28 * 28, 512), nn.ReLU(),
        nn.Linear(512, 10))

def make_cnn_slim():
    # the same CNN with a 32-unit dense layer: about half
    # the MLP's weights
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 32), nn.ReLU(),
        nn.Linear(32, 10))

mlp_wide, _ = fit(make_mlp_wide, epochs=5, verbose=False)
cnn_slim, _ = fit(make_cnn_slim, epochs=5, verbose=False)
print(f"{'model':10s} {'params':>8s} {'test acc':>9s}")
for name, model in [("MLP", mlp), ("wide MLP", mlp_wide),
                    ("slim CNN", cnn_slim), ("CNN", cnn)]:
    acc = accuracy(model, X_test, y_test)
    print(f"{name:10s} {n_params(model):8d} {acc:9.4f}")
```

~~~text
model        params  test acc
MLP          235146    0.8753
wide MLP     407050    0.8804
slim CNN     119530    0.9006
CNN          421642    0.9130
~~~

**Reading the table.** Giving the MLP as many weights as the CNN (407,050) changes almost nothing:
it stays within about half a point of the original MLP, near 0.88. Taking weights *away* from the
CNN -- the slim version has about half the MLP's parameters -- costs it about a point, and it still
beats both MLPs by two points or more. So the CNN does not win because it is bigger. It wins
because of *how* its weights are used: shared filters that look at neighborhoods, and pooling that
summarizes them. (We re-ran this comparison at different CPU thread counts and on a GPU: the
individual accuracies moved in the third decimal, but both CNNs always stayed two points or more
ahead of both MLPs, and the two MLPs always stayed within about half a point of each other.)

```python
epochs = range(1, 6)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(epochs, cnn_hist["loss"], "o-", label="CNN")
ax1.plot(epochs, mlp_hist["loss"], "o-", label="MLP")
ax1.set_xticks(list(epochs))
ax1.set_xlabel("epoch")
ax1.set_ylabel("training loss")
ax1.set_title("Training loss")
ax1.legend()
for hist, label, color in [(cnn_hist, "CNN", "C0"),
                           (mlp_hist, "MLP", "C1")]:
    ax2.plot(epochs, hist["val_acc"], "o-", color=color,
             label=f"{label} validation")
    ax2.plot(epochs, hist["train_acc"], "--", color=color,
             label=f"{label} train")
ax2.set_xticks(list(epochs))
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.set_title("Train (dashed) vs validation (solid)")
ax2.legend()
plt.tight_layout()
plt.show()
```

**Reading the curves.** On the left, the CNN's training loss is lower than the MLP's after every
epoch. On the right, the CNN's validation accuracy (solid blue) sits above the MLP's (solid orange)
at every epoch, by about three to five points. Both validation curves are still rising at epoch 5,
so this plot says the CNN learns faster and better on these images, not that the MLP has hit a
ceiling. Each dashed train line sits only a little above
its solid validation line: after 5 epochs neither model is overfitting much yet (Exercise 1 trains
longer and watches that gap grow).

Finally, look at what the CNN still gets wrong:

```python
cnn_pred = predict(cnn, X_test)
show_predictions(X_test, y_test, cnn_pred, grid=(2, 5))
show_predictions(X_test, y_test, cnn_pred, grid=(2, 5),
                 which="wrong", seed=1)

wrong = cnn_pred != y_test
print(f"CNN misses: {int(wrong.sum())} of {len(y_test)}")
counts = torch.bincount(y_test[wrong], minlength=10).tolist()
for name, c in sorted(zip(CLASSES, counts), key=lambda t: -t[1]):
    print(f"  true {name:12s} {c:4d}")
```

~~~text
CNN misses: 870 of 10000
  true Shirt         258
  true Coat          183
  true T-shirt/top   135
  true Pullover      135
  true Ankle boot     47
  true Dress          31
  true Bag            27
  true Trouser        23
  true Sneaker        17
  true Sandal         14
~~~

**Reading the mistakes.** The top grid shows five correct predictions and five misses; the bottom
grid shows misses only. Almost every miss is one garment taken for another: a coat called a
pullover, a shirt called a T-shirt, a long dark top called a dress. The count confirms it: the four
upper-body classes from Step 1 -- shirt, coat, T-shirt/top and pullover -- account for 711 of the 870
misses, more than 80%, with **shirt** alone at 258. Footwear and trousers are rarely confused. At
28x28 pixels in grayscale, a shirt and a T-shirt can share almost the same outline, and some of these
images are hard for a person to call too. Better architectures (L19) narrow this gap, but these four
classes are where Fashion-MNIST classifiers typically lose most of their accuracy.

## Step 7: Inside the Trained CNN

What has the network learned? Start with filters you can read: four hand-set 3x3 kernels applied to
one real test image, an ankle boot.

```python
img = X_test[:1].cpu()                  # [1, 1, 28, 28]
hand_kernels = {
    "vertical edge":   [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    "horizontal edge": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    "diagonal edge":   [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]],
    "blur":            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
}
fig, axes = plt.subplots(1, 5, figsize=(12, 2.8))
axes[0].imshow(img[0, 0], cmap="gray")
axes[0].set_title(f"input: {CLASSES[int(y_test[0])]}")
for ax, (name, k) in zip(axes[1:], hand_kernels.items()):
    k = torch.tensor(k, dtype=torch.float32).reshape(1, 1, 3, 3)
    if name == "blur":
        ax.imshow(F.conv2d(img, k / 9, padding=1)[0, 0],
                  cmap="gray")
    else:
        # symmetric color limits: zero is exactly mid-gray
        fmap = F.conv2d(img, k, padding=1)[0, 0]
        v = fmap.abs().max()
        ax.imshow(fmap, cmap="gray", vmin=-v, vmax=v)
    ax.set_title(name)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

In the three edge maps, mid-gray is zero, white a strong positive response and black a strong
negative one. The **vertical-edge** kernel lights up the near-vertical boundaries -- the slope of
the upper and the back of the shaft -- and stays gray along the flat top of the sole. The
**horizontal-edge** kernel does the opposite: the top of the sole is a bright band and its
underside a dark one, because those edges run horizontally. The **diagonal** kernel draws its
brightest line along the slanted front of the upper, the one edge that runs in its own direction;
it also catches the sole a little, since the sole's edges are partly diagonal to it. The **blur**
kernel has no negative weights, so its map has no negative values (black is zero there): it just
averages each 3x3 neighborhood, giving a softer copy of the boot. Same image, four filters, four
different pictures: each feature map shows where *its* pattern occurs.

Now the real thing. The first convolutional layer of the trained CNN has 32 filters, each a 3x3 grid
of learned weights. Below, every filter is drawn **directly above the feature map it produces** on the
same boot (after the ReLU). Filter weights are colored red for positive and blue for negative, with
white at zero, so you can see which way each one is tilted.

```python
w = cnn[0].weight.detach().cpu()[:, 0]          # [32, 3, 3]
with torch.no_grad():
    maps1 = F.relu(cnn[0](X_test[:1]))[0].cpu()  # [32, 28, 28]
fig, axes = plt.subplots(8, 8, figsize=(10, 11))
for k in range(32):
    row, col = 2 * (k // 8), k % 8
    v = w[k].abs().max()
    axes[row, col].imshow(w[k], cmap="RdBu_r", vmin=-v, vmax=v)
    axes[row, col].set_title(f"filter {k}", fontsize=7)
    axes[row + 1, col].imshow(maps1[k], cmap="gray")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()
```

A 3x3 grid of learned weights is hard to read on its own -- most filters are a mix of positive and
negative entries with no obvious picture in them. The feature maps underneath make each filter's job
visible:

- **Many filters trace edges.** Several maps show the boot's outline, the line of the sole, or the
  stitching on the upper as thin bright lines -- learned versions of the hand-set edge kernels above,
  each tuned to a different direction or contrast.
- **A few respond to the background.** Some maps are bright everywhere *except* the boot, which shows
  up as a dark silhouette: those filters fire on the plain dark surround.
- **Some respond to the body of the shoe**, lighting up the bright area of the upper rather than its
  edges.
- **Some are almost silent on this image.** Their pattern is simply not present in an ankle boot;
  they may fire strongly on a shirt or a bag.

On a GPU, or with a different seed, the learned filters come out somewhat different, so your figure
may not match this one filter for filter; look for the same *kinds* of maps -- edges, background,
filled regions, near-silent ones. None of these filters was designed: backprop found them because
they help tell the ten classes apart.

The second convolutional layer works on the first layer's 32 pooled maps and produces 64 maps of
14x14. Here are the first 16 for the same boot:

```python
with torch.no_grad():
    maps2 = cnn[:5](X_test[:1])[0].cpu()  # conv1-relu-pool-conv2-relu
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for k, ax in enumerate(axes.ravel()):
    ax.imshow(maps2[k], cmap="gray")
    ax.set_title(f"map {k}", fontsize=7)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

These maps are coarser (14x14, after one pooling) and less like line drawings. Each second-layer
filter combines all 32 first-layer maps, so its map tends to light up a **region** of the boot rather
than a single edge: some respond along the sole, some along the diagonal of the upper or around the
ankle opening, some to the area above the shoe, and one is dark on this image. This is the start of
the **feature hierarchy** from the concept note: the first layer finds edges, the second combines
them into larger parts. With only two convolutional layers on 28x28 images the hierarchy stops here;
the deeper networks of L19 carry it on to whole objects.

## Your Turn

### Exercise 1 -- Regularize the CNN (L17 tricks), and watch the gap

In Step 6 the CNN trained for only 5 epochs, before it had much chance to overfit. Train it longer
and see what L17's regularizers do. Build a regularized CNN with `nn.BatchNorm2d` after each
convolution and `nn.Dropout(0.5)` right after `nn.Flatten()`, then train **both** the plain CNN and
the regularized one for **20 epochs** and plot each model's train and validation accuracy per epoch.
How big is the gap between train and validation accuracy for each model at epoch 20, and does
validation accuracy keep improving?

**Hint:** insert `nn.BatchNorm2d(32)` after the first conv and `nn.BatchNorm2d(64)` after the second
(before each ReLU), and `nn.Dropout(0.5)` after `nn.Flatten()`. Then
`plain, plain_hist = fit(make_cnn, epochs=20, verbose=False)` and the same for your model; the
`train_acc` and `val_acc` lists in each history are what you plot (reuse the right-hand panel code
from Step 6, with `range(1, 21)`). Twenty epochs for two models takes about 2 minutes on a T4 GPU
and around 10 minutes or more on a CPU -- this is the cell to run on the GPU.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
              ----------- epoch 20 -----------
model          train    val     gap     best val      test
plain CNN      0.993   0.920  +0.073   0.927 (ep 7)  0.9147
BN + Dropout   0.980   0.931  +0.049   0.931 (ep 20) 0.9265
~~~

The regularized model is `nn.Sequential(nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
nn.MaxPool2d(2), nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
nn.Flatten(), nn.Dropout(0.5), nn.Linear(64*7*7,128), nn.ReLU(), nn.Linear(128,10))`. The table
above summarizes the two training histories; your own printout will be laid out differently, and the
numbers are what to compare.

In the plot, the **plain CNN's** train accuracy (dashed) climbs to about 0.99 -- it is close to
memorizing the training set -- while its validation accuracy (solid) peaks around epoch 7 and then
drifts sideways near 0.92. Its train-validation gap grows steadily, from under a point at epoch 1 to
about 7 points at epoch 20: classic overfitting (L09). The **regularized CNN's** train accuracy
climbs more slowly, because dropout keeps switching off half the dense layer's inputs during
training, and its gap grows to only about 5 points; its validation accuracy is still creeping up at
epoch 20. The takeaway is the **gap**: regularization made the network memorize less. In this run
that also bought about a point of test accuracy (0.927 against 0.915). We repeated the comparison
with three different seeds on a GPU: the regularized gap was about two points smaller every time,
and its test accuracy about a point higher every time. After only 5 epochs (Step 6's length), by
contrast, the two models' validation accuracies were the same within run-to-run noise: the plain CNN
had not started to overfit yet, so there was nothing for regularization to fix.
</details>

### Exercise 2 -- Average pooling instead of max

Swap both `nn.MaxPool2d(2)` for `nn.AvgPool2d(2)` in the plain CNN, train it for 5 epochs with `fit`,
and compare its test accuracy with the max-pooling CNN's. Which pooling keeps the sharper features
here?

**Hint:** copy `make_cnn` into a `make_cnn_avg` and replace the two `nn.MaxPool2d(2)` with
`nn.AvgPool2d(2)`; then `fit(make_cnn_avg, epochs=5)` and `accuracy(..., X_test, y_test)`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
  epoch  1  loss 0.527  train 0.864  val 0.861
  epoch  2  loss 0.344  train 0.891  val 0.887
  epoch  3  loss 0.295  train 0.888  val 0.877
  epoch  4  loss 0.265  train 0.904  val 0.893
  epoch  5  loss 0.242  train 0.920  val 0.906
CNN (AvgPool) test accuracy: 0.9034
~~~

Average pooling reaches about 0.903, a point below max pooling's 0.913 in Step 6. Max pooling keeps the
strongest activation in each window -- the clearest evidence that a feature is present -- while
averaging dilutes it with the weaker values around it. The difference is small, though: across three
seeds on a GPU, max pooling was ahead by about a point twice and tied once. Max pooling is the more
common default, but here it is a modest edge, not a rule.
</details>

## Summary

- A **convolution** is multiply-and-sum: a small **kernel** slid over each patch makes a **feature
  map**; `F.conv2d` matched our by-hand `convolve2D` exactly, and the output size follows
  `floor((n + 2p - k) / s) + 1` (padding 1 keeps the size, stride 2 halves it).
- A filter is a **pattern detector** -- the vertical-edge kernel fired on edges (+/-) and was silent on
  flat regions -- and the same filter is reused everywhere (**parameter sharing**), so a pattern is
  found wherever it appears. **Pooling** summarizes each 2x2 window (max 54, average 29.5 for the
  top-left window), halves height and width, and has **no weights**.
- A CNN stacks `Conv -> ReLU -> Pool` blocks (spatial trace **28 -> 14 -> 7**), then
  `Flatten -> Dense -> logits`; a conv layer has `(C_in * k * k + 1) * C_out` parameters -- **320** and
  **18,496** here -- while the first dense layer holds 401,536 of the 421,642.
- On the same Fashion-MNIST images the CNN reached about **0.91** test accuracy against the MLP's
  **0.88**, ahead at every epoch. It is not a matter of size: an MLP with the CNN's parameter count
  stayed near 0.88, and a CNN with half the MLP's parameters still reached about 0.90. Most of the
  CNN's remaining mistakes are between shirt, coat, T-shirt/top and pullover.
- Inside the trained CNN, first-layer filters respond to edges, the background, filled regions or
  nothing at all on a given image; second-layer maps respond to larger parts -- the start of the
  feature hierarchy.
- Trained for 20 epochs, the plain CNN overfits (train-validation gap about 7 points); L17's
  **BatchNorm + dropout** kept the gap near 5 points. Max pooling edged out average pooling by about a
  point, or tied.
- Next (L19): assembling these layers into famous architectures (ResNet's residual connections) and
  reusing models pretrained on millions of images -- **transfer learning**.
