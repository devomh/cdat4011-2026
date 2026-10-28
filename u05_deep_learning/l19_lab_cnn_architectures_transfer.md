---
title: "Lab: Transfer Learning with a Pretrained ResNet"
unit: "V"
lesson: "19"
type: lab
tags: [deep-learning, pytorch, resnet, residual-connections, transfer-learning, fine-tuning, torchvision, cifar-10]
difficulty: intermediate
duration: "60 mins"
---

**Goal:** reuse a network somebody else trained on a million images. You will load a **pretrained
ResNet18**, build a **residual block** by hand and then train a 19-layer CNN with and without the skip
connection to see what it buys, then do **transfer learning** -- freeze the backbone, train a tiny head
-- and watch it crush every CNN trained **from scratch** on the same little CIFAR-10 dataset. Everything
runs on CPU: about two minutes on a fast laptop, up to five on Colab's CPU (the one-time downloads aside). Pairs with the concept note
[CNN Architectures & Transfer Learning](l19_concept_cnn_architectures_transfer.qmd).

> **Previously:** L18 -- Convolutional Neural Networks (build one from scratch)  |  **Next:** L20 -- Autoencoders & Representation Learning

> This page is the read-only view. To run the lab, open the notebook (`l19_lab_cnn_architectures_transfer.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u05_deep_learning/l19_lab_cnn_architectures_transfer.ipynb)

## Scenario

You have only about a hundred labeled images per class -- far too few to train a deep CNN from scratch.
But a ResNet18 has already been trained on ImageNet's 1.2 million photos and knows what edges,
textures, and object-parts look like. We will reuse that knowledge: freeze its convolutional
**backbone**, treat it as a fixed feature extractor, and train a small classifier **head** on
CIFAR-10's ten object classes. Then we will train CNNs from scratch on the same data -- a shallow one,
and two deep ones that differ only in their skip connections -- and measure how badly they lose.

## Setup

Two cells: the first installs, the second imports, seeds, loads a small CIFAR-10 subset, and defines
helpers. CIFAR-10 images are only 32x32 pixels; we resize them to 96x96 and normalize with the
**ImageNet** statistics the pretrained model expects.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships these, so this is effectively a no-op there.
%pip install -q torch torchvision scikit-learn matplotlib
# local, in a terminal (not in the notebook):  uv add torch torchvision scikit-learn matplotlib
```

```python
# Setup, cell 2 of 2 -- IMPORTS + SEEDS + DATA + HELPERS (safe to re-run without re-installing)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset

torch.manual_seed(42); np.random.seed(42)

# A pretrained model expects its training-time preprocessing: ImageNet normalization, here at 96x96.
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
tfm = transforms.Compose([transforms.Resize(96), transforms.ToTensor(),
                          transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

train_full = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=tfm)
test_full  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tfm)
CLASSES = train_full.classes

def balanced_subset(ds, per_class, seed=42):
    """A seed-locked subset with `per_class` images from each of the 10 classes (keeps the lab fast)."""
    rng = np.random.RandomState(seed); targets = np.array(ds.targets); idx = []
    for c in range(10):
        idx.extend(rng.choice(np.where(targets == c)[0], per_class, replace=False))
    rng.shuffle(idx)
    return Subset(ds, list(idx))

train_set = balanced_subset(train_full, 100)   # 1000 images: 100 per class
test_set  = balanced_subset(test_full, 100)    # 1000 test images
test_loader = DataLoader(test_set, batch_size=256)

def loader_accuracy(model, loader):
    """Top-1 accuracy of a model over a DataLoader."""
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item(); total += len(yb)
    return correct / total

print(f"train {len(train_set)} / test {len(test_set)} images; {len(CLASSES)} classes")
print("classes:", ", ".join(CLASSES))
```

~~~text
train 1000 / test 1000 images; 10 classes
classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
~~~

## Step 1: Look at the Data

Before modeling, look at what we are classifying: one test image from each class.

```python
def to_rgb(img):
    """Undo the ImageNet normalization so a [3, H, W] tensor displays as an RGB picture."""
    rgb = img.permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN)
    return rgb.clamp(0, 1)

# The first test image of each class, found by index (no random sampling).
test_labels = [test_full.targets[i] for i in test_set.indices]
first_of_class = [test_labels.index(c) for c in range(10)]

plt.figure(figsize=(12, 1.8))
for c, i in enumerate(first_of_class):
    plt.subplot(1, 10, c + 1); plt.imshow(to_rgb(test_set[i][0]))
    plt.xticks([]); plt.yticks([]); plt.title(CLASSES[c], fontsize=8)
plt.show()

print("stored size:", train_full.data.shape[1:3], "-> model input:", tuple(test_set[0][0].shape))
```

~~~text
stored size: (32, 32) -> model input: (3, 96, 96)
~~~

The pictures look soft because each one is stored at only **32x32** pixels and blown up to 96x96. The
network sees exactly these blurry images. ResNet18 learned its features on ImageNet photos of about
224x224, so this is a real test of how well those features transfer.

## Step 2: Meet a Pretrained Network

Load ResNet18 with its ImageNet-trained weights. It is built from four stages of **residual blocks**
(`layer1`..`layer4`) and a classifier **head** (`fc`) that outputs 1000 ImageNet classes:

```python
net = resnet18(weights=ResNet18_Weights.DEFAULT)
print("ResNet18 total parameters:", sum(p.numel() for p in net.parameters()))
print("the classifier head:", net.fc)
```

~~~text
ResNet18 total parameters: 11689512
the classifier head: Linear(in_features=512, out_features=1000, bias=True)
~~~

To reuse it, split the network into a **backbone** (everything except the head) and a fresh **head**
for our 10 classes, and **freeze** the backbone so gradient descent leaves its weights fixed:

```python
backbone = nn.Sequential(*list(net.children())[:-1])   # all of ResNet18 except fc -> outputs [N, 512, 1, 1]
for p in backbone.parameters():
    p.requires_grad = False                            # freeze: these weights will not be updated

head = nn.Linear(512, 10)                              # a new, trainable head for CIFAR-10's 10 classes
print("trainable parameters after freezing the backbone:", sum(p.numel() for p in head.parameters()))
```

~~~text
trainable parameters after freezing the backbone: 5130
~~~

We went from 11.7 million parameters to **5,130** trainable ones -- only the new head learns. The
backbone is now a fixed function from an image to 512 features.

## Step 3: A Residual Block by Hand

What makes a network like ResNet trainable at 18, 50, or 150 layers is the **skip connection**. Build
one block -- two convolutions plus a shortcut that adds the input back -- and watch what it does:

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)          # the skip connection: add the input back before the final ReLU

torch.manual_seed(0)
block = ResidualBlock(8)
x = torch.randn(1, 8, 16, 16)
print("input shape:", tuple(x.shape), "-> output shape:", tuple(block(x).shape))

# Zero the block's conv weights so F(x) = 0; the output should then be exactly relu(x).
for p in block.parameters():
    nn.init.zeros_(p)
print("with F(x)=0, output == relu(x)?", torch.allclose(block(x), torch.relu(x)))
```

~~~text
input shape: (1, 8, 16, 16) -> output shape: (1, 8, 16, 16)
with F(x)=0, output == relu(x)? True
~~~

The block keeps the shape, so blocks can be stacked indefinitely. And when its convolutions output
nothing (`F(x) = 0`), the skip passes the input straight through -- the block falls back to the
**identity**. That is why adding residual blocks never makes a network harder to train: in the worst
case a block does nothing, instead of corrupting the signal. Real ResNet blocks add batch-norm and sit inside the `net` you loaded
in Step 2. In Step 6 you will train a deep CNN built from this very class.

## Step 4: Transfer Learning = Feature Extraction

Run every image through the frozen backbone **once** to get its 512 features, then train the small head
on those features. Because the backbone never changes, we can extract the features ahead of time and
the head trains in a blink:

```python
def extract_features(dataset):
    """Run the frozen backbone over a dataset -> (features [N, 512], labels [N])."""
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for xb, yb in loader:
            feats.append(backbone(xb).flatten(1)); labels.append(yb)
    return torch.cat(feats), torch.cat(labels)

Xtr, ytr = extract_features(train_set)
Xte, yte = extract_features(test_set)
print("feature matrix:", tuple(Xtr.shape))

torch.manual_seed(42)
head = nn.Linear(512, 10)
opt = torch.optim.Adam(head.parameters(), lr=1e-3); loss_fn = nn.CrossEntropyLoss()
for step in range(60):
    opt.zero_grad(); loss = loss_fn(head(Xtr), ytr); loss.backward(); opt.step()
    if step % 20 == 0:
        print(f"step {step:2d}  loss {loss.item():.3f}")
transfer_acc = (head(Xte).argmax(1) == yte).float().mean().item()
print(f"transfer-learning test accuracy: {transfer_acc:.4f}")
```

~~~text
feature matrix: (1000, 512)
step  0  loss 2.822
step 20  loss 0.775
step 40  loss 0.448
transfer-learning test accuracy: 0.7310
~~~

With **1000 training images** and a head of 5,130 weights, transfer learning reaches about **0.73**
accuracy on ten classes -- and it took seconds, because the expensive part (the backbone) was reused,
not trained. Here is the model at work on the first eight test images: its prediction above each
image, the true class in parentheses, and the misses in red.

```python
xb, yb = next(iter(test_loader))
with torch.no_grad():
    pred = head(backbone(xb[:8]).flatten(1)).argmax(1)
plt.figure(figsize=(10, 2.2))
for i in range(8):
    guess, true = CLASSES[pred[i]], CLASSES[yb[i]]
    plt.subplot(1, 8, i + 1); plt.imshow(to_rgb(xb[i])); plt.xticks([]); plt.yticks([])
    plt.title(f"{guess}\n({true})", fontsize=8, color="black" if guess == true else "red")
plt.show()
```

Six of the eight are right. Both misses are called "cat": a fluffy white dog and a truck seen head-on --
at 32x32 pixels, not unreasonable guesses.

## Step 5: Why Bother? Train From Scratch Instead

Now build an equivalent CNN -- the L18 recipe, sized for 96x96 inputs -- and train it on the **same**
1000 images, with no pretrained weights. We wrap the training loop in a function, because Step 6 reuses
it for two more networks:

```python
def make_scratch_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 96 -> 48
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 48 -> 24
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 24 -> 12
        nn.Flatten(),
        nn.Linear(64 * 12 * 12, 128), nn.ReLU(),                       # 64*12*12 = 9216
        nn.Linear(128, 10),
    )

def train_scratch(model, epochs=12):
    """Train a model on train_set (Adam, lr=1e-3, batches of 64); return the per-epoch training loss."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_set, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    curve = []
    for epoch in range(epochs):
        model.train(); running = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            running += loss.item() * len(xb)
        curve.append(running / len(train_set))
    return curve

torch.manual_seed(42)
scratch = make_scratch_cnn()
curve = train_scratch(scratch)
scratch_acc = loader_accuracy(scratch, test_loader)
print(f"from-scratch CNN test accuracy: {scratch_acc:.4f}")
print("loss curve:", " ".join(f"{l:.2f}" for l in curve))
```

~~~text
from-scratch CNN test accuracy: 0.4190
loss curve: 2.16 1.84 1.60 1.37 1.10 0.82 0.64 0.40 0.24 0.14 0.08 0.06
~~~

The from-scratch CNN reaches only about **0.42** -- a gap of roughly **0.31** below transfer learning,
on identical data. (Training from scratch is sensitive to tiny rounding differences between machines, so
you may see anything from about 0.41 to 0.44; the transfer model's 0.7310 does not move.) Its training loss falls to about 0.06, so it has essentially memorized the 1000 training
images, but what it memorized does not carry over to new ones. It is not a bad network; 1000 images is
simply far too few to learn good visual features from nothing. Transfer learning skips that bottleneck
by reusing features ImageNet already paid for.

Before going deeper, read the from-scratch CNN's flattened dimension straight off its spatial trace --
the one number you must get right to wire conv layers into a dense head:

```python
# Your turn (warm-up): from the 96 -> 48 -> 24 -> 12 trace, predict the flattened size that feeds the
# first dense layer, then check it. Uncomment and fill the blank:
# flat_dim = ____                              # 64 feature maps of 12 x 12
# assert flat_dim == scratch[10].in_features   # scratch[10] is the first nn.Linear
# print("flattened dimension:", flat_dim)
```

The blank is `64 * 12 * 12`, which is `9216` -- the `in_features` of the first `nn.Linear`.

## Step 6: Do Skips Matter? A Deep Plain CNN vs a Deep Residual CNN

If more layers meant more power, we could make the from-scratch CNN deeper. Let's add **8 blocks --
16 more convolutions** -- at its 12x12 stage, for 19 conv layers in all, and build it twice: once from a
`PlainBlock` (two convolutions, nothing added back) and once from Step 3's `ResidualBlock`. The two
networks have the same layers, the same parameters, and -- seeded the same way -- the same starting
weights. The **only** difference is the `+ x`.

```python
class PlainBlock(nn.Module):
    """ResidualBlock without the skip: the same two convolutions, nothing added back."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out)              # no "+ x"

def make_deep_cnn(block):
    """The Step 5 CNN with 8 extra blocks (16 more conv layers) at the 12x12 stage."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 96 -> 48
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 48 -> 24
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 24 -> 12
        *[block(64) for _ in range(8)],                                # 12 -> 12, 8 times
        nn.Flatten(),
        nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )

torch.manual_seed(42); deep_plain = make_deep_cnn(PlainBlock)
torch.manual_seed(42); deep_res = make_deep_cnn(ResidualBlock)
print("parameters, plain vs residual:", sum(p.numel() for p in deep_plain.parameters()),
      sum(p.numel() for p in deep_res.parameters()))

plain_curve = train_scratch(deep_plain)        # about a minute each on a CPU
res_curve = train_scratch(deep_res)
deep_plain_acc = loader_accuracy(deep_plain, test_loader)
deep_res_acc = loader_accuracy(deep_res, test_loader)
print(f"deep plain CNN:    test accuracy {deep_plain_acc:.4f}, final training loss {plain_curve[-1]:.2f}")
print(f"deep residual CNN: test accuracy {deep_res_acc:.4f}, final training loss {res_curve[-1]:.2f}")
```

~~~text
parameters, plain vs residual: 1828234 1828234
deep plain CNN:    test accuracy 0.1000, final training loss 2.30
deep residual CNN: test accuracy 0.3640, final training loss 0.38
~~~

```python
epochs = range(1, 13)
plt.figure(figsize=(6, 4))
plt.plot(epochs, curve, marker="o", label="shallow CNN (Step 5)")
plt.plot(epochs, plain_curve, marker="s", label="deep plain CNN")
plt.plot(epochs, res_curve, marker="^", label="deep residual CNN")
plt.axhline(np.log(10), color="gray", ls=":", label="guessing: ln(10) = 2.30")
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.xticks(list(epochs))
plt.title("From scratch on 1000 CIFAR-10 images"); plt.legend(); plt.show()
```

The deep plain CNN never learns anything. Its training loss stays at about 2.30 from the first epoch to
the last -- essentially ln(10), the loss of a model that spreads its bets evenly over 10 classes -- and
its test accuracy is 0.10, pure chance. Sixteen more convolutions made it worse than the shallow CNN, and
not by overfitting: it cannot even fit its training data. That is the **degradation problem** from the
concept note. The deep residual CNN has the same layers, the same parameters, and the same starting
weights, and it trains: its loss falls steadily, to roughly 0.3-0.4 by epoch 12. The only change is the
`+ x`, which gives the signal -- and its gradient on the way back -- a path around every block.

Now look at the residual CNN's test accuracy: about 0.36, **below** the shallow CNN's 0.42. Part of
that is time -- after 12 epochs its training loss is still far above the shallow CNN's 0.06 -- but the
shallow CNN shows where this road ends: it memorized all 1000 training images and still stops near 0.42.
Skip connections make a deep network *trainable*; they cannot supply the data needed to learn good
visual features. Depth needs skips; little data needs pretrained features. (On another machine the
residual CNN may land anywhere from about 0.35 to 0.38 in accuracy and 0.29 to 0.38 in final loss,
always below the shallow CNN; the plain CNN's 2.30 and 0.10 will not change.)

## Step 7: All Four Models Side by Side

```python
def n_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

rows = [
    ("shallow CNN, from scratch", n_params(scratch), n_params(scratch), scratch_acc),
    ("deep plain CNN, from scratch", n_params(deep_plain), n_params(deep_plain), deep_plain_acc),
    ("deep residual CNN, from scratch", n_params(deep_res), n_params(deep_res), deep_res_acc),
    ("ResNet18 backbone + new head", n_params(backbone) + n_params(head), n_params(head), transfer_acc),
]
print(f"{'model':32s} {'params':>10s} {'trainable':>10s} {'test acc':>9s}")
for name, total, trainable, acc in rows:
    print(f"{name:32s} {total:>10,d} {trainable:>10,d} {acc:>9.4f}")
```

~~~text
model                                params  trainable  test acc
shallow CNN, from scratch         1,237,386  1,237,386    0.4190
deep plain CNN, from scratch      1,828,234  1,828,234    0.1000
deep residual CNN, from scratch   1,828,234  1,828,234    0.3640
ResNet18 backbone + new head     11,181,642      5,130    0.7310
~~~

Read the table from the bottom. The transfer model is by far the biggest network, yet it trains only
5,130 weights -- and it wins by about 0.3. Every model trained from scratch, shallow or deep, with or
without skips, lands between chance and the mid-0.40s. Adding layers did not help the from-scratch models;
reusing features learned on a million images did.

## Squeezing Out More Accuracy (a Taste)

Can we do better than 0.73, or rescue the from-scratch CNN? Here is what a few standard techniques do
on this lab's data. These numbers were measured with the lab's own code and recipe (seed 42, the same
1000/1000 images unless noted); they are not computed in this notebook, to keep it fast.

| What we changed | Test accuracy | Final training loss | Course link |
|---|---|---|---|
| Transfer: inputs 160x160 instead of 96x96 | 0.731 -> **0.777** | | ImageNet size, concept note |
| Transfer: inputs 224x224, ImageNet's own size | 0.731 -> **0.779** | | ImageNet size, concept note |
| Transfer: 500 images per class instead of 100 | 0.731 -> **0.769** | | more data |
| From scratch + data augmentation, 12 epochs | 0.419 -> 0.434 | 0.06 -> 1.22 | new here |
| From scratch + data augmentation, 24 epochs | 0.419 -> 0.449 | 0.06 -> 0.72 | new here |
| From scratch + BatchNorm + Dropout(0.5) | 0.419 -> 0.405 | 0.06 -> 1.63 | L17 |

Data augmentation, the one technique no earlier lesson showed, is a single extra transform applied to
the **training** images only, so each epoch sees a slightly different crop or mirror image of every
picture:

~~~python
aug = transforms.Compose([
    transforms.Resize(96),
    transforms.RandomCrop(96, padding=8),   # a random 96x96 window
    transforms.RandomHorizontalFlip(),      # mirror half the time
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
~~~

Two patterns stand out. **For transfer, the big wins come from the inputs and the data**: giving the
pretrained network images closer to the size it learned on, or five times more labeled images, adds
4 to 5 points. **For the from-scratch CNN, regularization changes how it learns but not how well**:
augmentation and dropout stop it from memorizing (its training loss stays between 0.7 and 1.6 instead of
falling to 0.06), yet its test accuracy stays in the same 0.38-0.45 band. We re-ran every row at
different CPU thread counts: the transfer rows did not move, while the from-scratch rows moved by up to
about 0.03 -- as much as the differences between them. On a 1000-image test set, gaps that small are
noise. Nothing here closes the gap between from-scratch (about 0.4) and transfer (about 0.75).

Other techniques you have met, or will meet, belong on the same list: **weight decay** and **early
stopping** (L10), a **learning-rate schedule** (L17), a **bigger or newer pretrained backbone** (see the
model hubs in the concept note), **fine-tuning** (Exercise 1 below), and **ensembles** of several models
(L12-L13). Label smoothing and mixup are two more regularizers that the CDAT 4013 lab covers.

## Your Turn

### Exercise 1 -- Fine-tuning

Feature extraction froze the *entire* backbone. **Fine-tuning** unfreezes the top block to let backprop
adapt the high-level features. Reload a fresh `resnet18(weights=ResNet18_Weights.DEFAULT)`, freeze it,
replace `fc` with `nn.Linear(512, 10)`, then set `requires_grad = True` on `model.layer4` and the new
`fc`. Train end-to-end on `train_set` for ~4 epochs at a **low** `lr=1e-4` and report test accuracy.

**Hint:** start with `torch.manual_seed(42)`, *then* reload
`resnet18(weights=ResNet18_Weights.DEFAULT)`, freeze it, set `model.fc = nn.Linear(512, 10)`, and
`for p in model.layer4.parameters(): p.requires_grad = True` -- seeding first fixes the new head's
initialization so your number matches the reveal exactly. Collect the trainable parameters with
`[p for p in model.parameters() if p.requires_grad]` and pass them to `Adam(..., lr=1e-4)`; reuse a
`DataLoader(train_set, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))` and
`loader_accuracy(...)`. Fine-tuning trains *through* the backbone, so it is slower than Step 4 (up to a
minute on a CPU).

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
fine-tuned (layer4 + head) test accuracy: 0.7090
~~~

Unfreezing `layer4` makes about **8.4 million** parameters trainable, and after 4 epochs at lr=1e-4 the
model reaches about **0.71** -- essentially the same as feature extraction's 0.73, **not better**. That
is the honest lesson: with only 100 images per class there is too little data to improve the pretrained
features, and fine-tuning risks disturbing them. Fine-tuning pays off when you have *more* data; with
little data, freezing everything (feature extraction) is the right call. One honest caveat: this
exercise takes a shortcut, training the brand-new head and `layer4` together from the start. The full
recipe (the concept note's row 3) trains the head first and only then unfreezes `layer4`, so the
pretrained features are never pushed around by a random head. With the `torch.manual_seed(42)` from the
hint you should get exactly this number; the takeaway is the comparison, not the last digit.
</details>

### Exercise 2 -- Does the Head Matter?

Step 4 put the smallest possible head on the frozen features: one `nn.Linear(512, 10)`. Many older
tutorials use a bigger head, such as 512 -> 256 -> 10. Try two alternatives on the cached features
`Xtr, ytr` from Step 4 and score each on `Xte, yte`. **First,** fit scikit-learn's
`LogisticRegression` (no neural network at all). **Second,** train a two-layer head
`nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10))` exactly as Step 4
trained its head. Compare both with the one-layer head's 0.7310, and count the parameters.

**Hint:** for the first, `from sklearn.linear_model import LogisticRegression`, then
`LogisticRegression(max_iter=1000).fit(Xtr.numpy(), ytr.numpy())` and compare
`clf.predict(Xte.numpy())` to `yte.numpy()` -- the logistic regression of Unit II (L04), now on features
a CNN extracted. For the second, call `torch.manual_seed(42)` *before* building the head (so your number
matches the reveal), reuse Step 4's loop (60 full-batch Adam steps at `lr=1e-3`), and call `.eval()` on
the head before scoring it, so dropout is switched off.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
logistic-regression head test accuracy: 0.7370
two-layer head (133,898 params) test accuracy: 0.7530
~~~

Both land within about 2 points of the one-layer head's 0.7310 -- and the two-layer head needs **26
times** its 5,130 parameters to get there. On a 1000-image test set one standard error is about 1.4
points, so a 2-point gap is barely outside the noise; averaged over five different seeds, the two-layer
head comes out only about 1 point ahead. The choice of classifier on top barely matters; the win lives in the **features** the pretrained backbone produced. "Transfer learning" is
mostly "let a pretrained CNN turn images into good features, then run any classifier you like" --
including the logistic regression you met in Unit II.

Why do older tutorials use a 256-unit hidden layer, then? Most come from the VGG era, when the head sat
on a *flattened* 7x7x512 feature map of 25,088 numbers. ResNet averages each feature map down to one
number first, so its 512 features are already compact, and its own head is a single linear layer -- the
standard default today. A hidden layer earns its keep with more data, or with images very different
from ImageNet.
</details>

## Summary

- **CIFAR-10** images are only 32x32; the lab enlarged them to 96x96 for a network that learned on
  ~224x224 **ImageNet** photos.
- A **pretrained** ResNet18 (11.7M parameters, an ImageNet head of 1000 classes) splits into a reusable
  conv **backbone** and a task-specific **head**. **Freezing** the backbone left only **5,130**
  trainable parameters.
- A **residual block** (two convs plus a skip) keeps its input shape and falls back to the **identity**
  when `F(x) = 0`. Stacked 8 deep, it trained; the same 19-layer CNN **without** skips stayed at
  training loss 2.30 and 0.10 test accuracy -- the **degradation problem**, an optimization failure.
- **Transfer learning** as feature extraction -- frozen backbone, trained head -- reached **0.73** on a
  1000-image CIFAR-10 subset. No CNN trained **from scratch** on the same data got past about **0.45**:
  skips make depth trainable, but only pretrained features beat the data shortage.
- **Fine-tuning** the top block did not beat feature extraction here (~0.71): with little data, freeze
  everything. The head barely matters either: a logistic-regression head (~0.74) and a two-layer head
  (~0.75) land within noise of the one-layer head -- the features carry the win.
- To push transfer further, feed it bigger inputs or more data (~0.77-0.78); regularizers curbed the
  from-scratch CNN's memorizing but did not close the gap.
- Next (L20): instead of reusing features learned from *labels*, **autoencoders** learn useful features
  with no labels at all.
