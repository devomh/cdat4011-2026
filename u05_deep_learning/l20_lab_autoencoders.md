---
title: "Lab: Autoencoders, from Scratch to Stable Diffusion"
unit: "V"
lesson: "20"
type: lab
tags: [deep-learning, pytorch, autoencoder, latent-space, denoising, variational-autoencoder, convolutional-autoencoder, anomaly-detection, stable-diffusion, latent-editing]
difficulty: intermediate
duration: "90 mins"
---

**Goal:** train networks with **no labels**, then use autoencoders that others trained. Part A builds
autoencoders from scratch on FashionMNIST: an MLP autoencoder, a look at its latent space in 2-D and
4-D, a plain autoencoder against a **variational** one, a denoiser, a **convolutional** autoencoder and
an anomaly detector. Part B loads two trained autoencoders: **Stable Diffusion's**, which squeezes a
photo 48-fold and gives it back sharp, and a face VAE whose latent space you steer with **sliders**.
Pairs with the concept note [Autoencoders & Representation Learning](l20_concept_autoencoders.qmd).

> **Previously:** L19 -- CNN Architectures & Transfer Learning (reuse labeled features)  |  **Next:** L21 -- Attention Mechanisms

> This page is the read-only view. To run the lab, open the notebook (`l20_lab_autoencoders.ipynb`) -- in Colab via the badge below, or locally. **Use a GPU runtime** (Runtime > Change runtime type > T4 GPU): Part A trains about a dozen models. Everything also runs on a CPU, just several times slower. The outputs shown on this page came from a CPU, so your numbers on a GPU or another machine will differ slightly.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u05_deep_learning/l20_lab_autoencoders.ipynb)

## Scenario

Nobody labeled your images, but you still want useful features. An autoencoder supervises itself: its
target is its own input. Force the signal through a code far too narrow to carry 784 pixels and the
network has to learn what *matters*. We will watch that code rebuild images, sort the classes on a map
it was never told about, clean up noise, and flag the odd one out. Then we switch from our small models
to the autoencoder inside Stable Diffusion, and to a face model you can edit with sliders.

## Setup

Two cells: the first installs, the second imports, picks the device, loads the data and defines the
training helpers.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships everything except possibly diffusers (Step 7).
%pip install -q torch torchvision scikit-learn scikit-image matplotlib ipywidgets safetensors diffusers
# local, in a terminal (not in the notebook):
#   uv add torch torchvision scikit-learn scikit-image matplotlib ipywidgets safetensors diffusers
```

```python
# Setup, cell 2 of 2 -- IMPORTS + DEVICE + DATA + HELPERS (safe to re-run)
import itertools
import json
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

# logistic regression on raw codes needs many iterations; the
# accuracies we report are stable, so silence the reminder
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    return ds.data.float().div(255).unsqueeze(1), ds.targets

X_train, y_train = to_tensors(train_raw)
X_test, y_test = to_tensors(test_raw)
X_train, X_test = X_train.to(device), X_test.to(device)

def mse_loss(model, x_in, x_target):
    """The plain reconstruction loss: mean squared error."""
    return F.mse_loss(model(x_in), x_target)

def fit_ae(make_model, epochs=10, seed=0, noise=0.0,
           X=None, loss_fn=mse_loss):
    """Build a fresh autoencoder under a fixed seed and train
    it with Adam to reconstruct X (default: X_train). With
    noise > 0 it sees a noisy copy but must output the clean
    image. Returns the model and the loss after each epoch."""
    X = X_train if X is None else X
    torch.manual_seed(seed)
    model = make_model().to(device)
    loader = DataLoader(
        TensorDataset(X), batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    curve = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (xb,) in loader:
            x_in = xb
            if noise > 0:
                x_in = (xb + noise * torch.randn_like(xb))
                x_in = x_in.clamp(0, 1)
            opt.zero_grad()
            loss = loss_fn(model, x_in, xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        curve.append(total / len(X))
    model.eval()
    return model, curve

def reconstruct(model, X):
    """Reconstructions of X, 2000 images at a time."""
    with torch.no_grad():
        return torch.cat([model(X[i:i + 2000])
                          for i in range(0, len(X), 2000)])

def encode(model, X):
    """Latent codes of X as a NumPy array (on the CPU)."""
    with torch.no_grad():
        return torch.cat([model.encoder(X[i:i + 2000])
                          for i in range(0, len(X), 2000)]).cpu().numpy()

def mse(a, b):
    return F.mse_loss(a, b).item()

def n_params(model):
    return sum(p.numel() for p in model.parameters())

def logreg_accuracy(Z_train, Z_test, n=10000):
    """Fit logistic regression on n training codes; test it."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z_train[:n], y_train[:n].numpy())
    return (clf.predict(Z_test) == y_test.numpy()).mean()

def show_rows(rows, titles, n=8):
    """One row of n images per entry in rows (each [N,1,28,28])."""
    fig, axes = plt.subplots(len(rows), n,
                             figsize=(1.3 * n, 1.45 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (imgs, title) in enumerate(zip(rows, titles)):
        imgs = imgs[:n].detach().cpu()
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].reshape(28, 28), cmap="gray_r",
                      vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(title, rotation=0, ha="right",
                              va="center", fontsize=9)
    fig.tight_layout()
    plt.show()

print("device:", device)
print(f"train {len(X_train)} / test {len(X_test)} images; "
      f"{len(CLASSES)} classes")
```

~~~text
device: cpu
train 60000 / test 10000 images; 10 classes
~~~

Two helpers do most of the work below:

- **`fit_ae(make_model, ...)`** builds a model under a fixed seed and trains it to reconstruct the
  training images. The loss is a function argument because it is the one thing Step 3 changes: by
  default it is `mse_loss`, the mean squared error between the output and the input itself -- **no
  labels anywhere**. Every call seeds itself, so re-running any training cell alone reproduces it.
- **`logreg_accuracy(Z_train, Z_test)`** measures how much class information a code holds: it fits
  L04's logistic regression on the codes of 10,000 training images (with their labels -- the only place
  labels appear, and only to *grade* the codes) and reports test accuracy.

The data is **Fashion-MNIST** (Zalando Research, MIT licence): 60,000 training and 10,000 test images of
clothing, 28x28 grayscale, 10 classes. All of it is used here; on a GPU each training run below takes
seconds.

## Part A -- Autoencoders from Scratch

### Step 1: An MLP Autoencoder

The concept note's definition, in code: an **encoder** $f_\theta$ that funnels 784 pixels down to a
32-number **code** $z$, and a **decoder** $g_\phi$ that expands the code back to 784 pixels. Because the
code is far smaller than the input, the network is **undercomplete**: it cannot copy, it must compress.

```python
class MLPAutoencoder(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)))

    def forward(self, x):
        return self.decoder(self.encoder(x))

ae, curve = fit_ae(lambda: MLPAutoencoder(latent=32))
print(f"parameters: {n_params(ae):,}")
print(f"training loss: epoch 1 {curve[0]:.4f} -> "
      f"epoch {len(curve)} {curve[-1]:.4f}")
print(f"test reconstruction MSE: "
      f"{mse(reconstruct(ae, X_test), X_test):.4f}")
```

~~~text
parameters: 439,728
training loss: epoch 1 0.0372 -> epoch 10 0.0124
test reconstruction MSE: 0.0125
~~~

The training loss falls from 0.037 to 0.012 in ten epochs, and the test images -- which the model
never trained on -- reconstruct just as well (MSE **0.0125**): the autoencoder learned clothing, not the
training set. Compare originals with their reconstructions:

```python
show_rows([X_test, reconstruct(ae, X_test[:8])],
          ["original", "from 32\nnumbers"])
```

Every garment is recognizable from 32 numbers -- the boot, the trousers, the sweaters -- but the
details 32 numbers cannot afford are gone: the "Lee" print, the stripes, the plaid. What survives is the
shape and the overall shade, the things most clothing images have in common. That is compression: keep
the structure, drop the rest.

### Step 2: The Latent Space, in 2-D and in 4-D

Shrink the code to **2** numbers and you can plot it. Train a 2-number autoencoder, encode every test
image, and color each point by its class -- a label the autoencoder never saw:

```python
ae2, _ = fit_ae(lambda: MLPAutoencoder(latent=2))
Z2_train, Z2_test = encode(ae2, X_train), encode(ae2, X_test)

def scatter_codes(Z, ax, title):
    for c in range(10):
        m = y_test.numpy() == c
        ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.5,
                   color=plt.cm.tab10(c), label=CLASSES[c])
    ax.set_title(title)
    ax.set_xlabel("code 1")
    ax.set_ylabel("code 2")

fig, ax = plt.subplots(figsize=(7, 5.5))
scatter_codes(Z2_test, ax, "2-D codes of 10,000 test images")
ax.legend(markerscale=4, fontsize=8, loc="center left",
          bbox_to_anchor=(1, 0.5))
plt.show()
acc2 = logreg_accuracy(Z2_train, Z2_test)
print(f"logistic regression on the 2 code numbers: {acc2:.3f}")
```

~~~text
logistic regression on the 2 code numbers: 0.582
~~~

With no labels, the classes still form regions: trousers (orange) on their own at the bottom right,
the footwear -- sandals, sneakers, ankle boots -- along the top, bags apart. The tops (T-shirt, pullover,
coat, shirt) pile up together on the left: with two numbers the code can say "a top" but not which one.
Logistic regression on the two numbers gets **0.582** of the test images right, almost six times chance
(0.10), from a code learned without a single label. (This number varies more than most from machine to
machine -- between about 0.58 and 0.67 in our runs -- because a 2-number code depends heavily on exactly
how training happened to lay the classes out.)

Now give the code **4** numbers. It can no longer be drawn in one picture, so draw all six pairs of
coordinates -- six 2-D **projections** of the same 4-D cloud -- with each coordinate's distribution per
class on the diagonal:

```python
ae4, _ = fit_ae(lambda: MLPAutoencoder(latent=4))
Z4_train, Z4_test = encode(ae4, X_train), encode(ae4, X_test)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, j in itertools.product(range(4), range(4)):
    ax = axes[i, j]
    for c in range(10):
        m = y_test.numpy() == c
        if i == j:
            ax.hist(Z4_test[m, i], bins=40, histtype="step",
                    color=plt.cm.tab10(c))
        else:
            ax.scatter(Z4_test[m, j], Z4_test[m, i], s=1,
                       alpha=0.3, color=plt.cm.tab10(c))
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_ylabel(f"code {i + 1}")
    if i == 3:
        ax.set_xlabel(f"code {j + 1}")
fig.suptitle("4-D codes: every pair of coordinates")
dots = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(c))
        for c in range(10)]
fig.legend(dots, CLASSES, loc="lower center", ncol=5, fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()

print(f"all 4 code numbers:   "
      f"{logreg_accuracy(Z4_train, Z4_test):.3f}")
for i, j in itertools.combinations(range(4), 2):
    acc = logreg_accuracy(Z4_train[:, [i, j]], Z4_test[:, [i, j]])
    print(f"only codes {i + 1} and {j + 1}: {acc:.3f}")
```

~~~text
all 4 code numbers:   0.734
only codes 1 and 2: 0.326
only codes 1 and 3: 0.361
only codes 1 and 4: 0.543
only codes 2 and 3: 0.496
only codes 2 and 4: 0.569
only codes 3 and 4: 0.561
~~~

With all four numbers the accuracy rises to **0.734**: more room, more class information. Now look at
the six projections. **None** of them scores as well as the dedicated 2-D code (0.326 to 0.569, against
0.582; this held in every run we tried), and the picture changes from panel to panel -- the trousers sit cleanly apart in "code 2 vs code
4" and are mixed in with the rest in "code 1 vs code 2". The 4-D autoencoder had no reason to pack its
information into any particular pair of axes, so every 2-D view is a shadow that hides part of it. Its
axes also come in no order: train it again from another seed and a different pair can look best. (PCA, in
L26, is the method whose axes are ranked by importance.)

### Step 3: Same Network, Different Loss -- a Plain Autoencoder vs a VAE

Step 2's 2-D codes spread over an arbitrary range, with gaps between the classes. What does the decoder
draw if you hand it a code from a gap? Decode a 15x15 grid of codes covering the cloud and see:

```python
def decode_grid(decoder, lo, hi, n=15):
    """Decode an n x n grid of 2-D codes spanning [lo, hi]."""
    a = np.linspace(lo[0], hi[0], n)
    b = np.linspace(hi[1], lo[1], n)   # top row = high code 2
    z = torch.tensor([[u, v] for v in b for u in a],
                     dtype=torch.float32, device=device)
    with torch.no_grad():
        imgs = decoder(z).reshape(n, n, 28, 28).cpu()
    return imgs.permute(0, 2, 1, 3).reshape(n * 28, n * 28)

lo2 = np.percentile(Z2_test, 1, axis=0)
hi2 = np.percentile(Z2_test, 99, axis=0)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
scatter_codes(Z2_test, axes[0], "plain autoencoder: codes")
axes[0].set_xlim(lo2[0], hi2[0])
axes[0].set_ylim(lo2[1], hi2[1])
axes[1].imshow(decode_grid(ae2.decoder, lo2, hi2), cmap="gray_r")
axes[1].set_title("decoded grid over the same square")
axes[1].axis("off")
plt.show()
print("code 1 range:", lo2[0].round(1), "to", hi2[0].round(1))
print("code 2 range:", lo2[1].round(1), "to", hi2[1].round(1))
```

~~~text
code 1 range: -8.0 to 8.2
code 2 range: -17.0 to 3.7
~~~

The plain autoencoder's codes cover an arbitrary, lopsided range (code 2 runs from -17.0 to 3.7):
nothing in its loss says where codes should go. Its decoded grid is clear where codes are dense -- tops,
trousers, boots, sneakers -- but in the empty middle-right of the scatter, where no image was encoded,
the decoder draws faint ghosts and near-blank tiles: it was never trained to draw anything there. That is
why you cannot make new images by picking codes for a plain autoencoder -- you do not know where the
valid ones are.

A **variational autoencoder** keeps the same layers but changes the encoder's output and the loss (concept
note). The encoder now outputs a mean `mu` and a log-variance `logvar` for each image; the code is
*sampled* as `mu + sigma * eps` (the reparameterization trick); and the loss adds the **KL term**, which
pulls every image's cloud of codes toward one shared standard normal:

```python
class MLPVAE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU())
        self.to_mu = nn.Linear(64, latent)
        self.to_logvar = nn.Linear(64, latent)
        self.decoder = MLPAutoencoder(latent).decoder

    def encoder(self, x):
        # the code of an image = the center of its cloud
        return self.to_mu(self.body(x))

    def forward(self, x):
        h = self.body(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.decoder(z), mu, logvar

def vae_loss(model, x_in, x_target, beta=1.0):
    """Reconstruction (BCE, summed per image) + beta * KL."""
    x_hat, mu, logvar = model(x_in)
    n = len(x_target)
    rec = F.binary_cross_entropy(x_hat, x_target,
                                 reduction="sum") / n
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()) / n
    return rec + beta * kl

vae, vae_curve = fit_ae(lambda: MLPVAE(latent=2),
                        loss_fn=vae_loss)
Zv_train, Zv_test = encode(vae, X_train), encode(vae, X_test)
print(f"VAE loss: epoch 1 {vae_curve[0]:.1f} -> "
      f"epoch {len(vae_curve)} {vae_curve[-1]:.1f}")
print("code range:", Zv_test.min(0).round(1), "to",
      Zv_test.max(0).round(1))
print(f"logistic regression on the 2 VAE code numbers: "
      f"{logreg_accuracy(Zv_train, Zv_test):.3f}")
```

~~~text
VAE loss: epoch 1 299.3 -> epoch 10 260.2
code range: [-3.8 -2.6] to [3.1 2.2]
logistic regression on the 2 VAE code numbers: 0.651
~~~

The loss values are larger than Step 2's only because this loss *sums* over the 784 pixels and adds
the KL term. Look at the code range instead: every code now lies between about -3.8 and 3.1, packed
around the origin as the KL term demands -- and the two numbers still carry the classes (logistic
regression **0.651**, in the same range as the plain code's): the KL term moved the codes, it did not
erase what they say. Decode the same kind of grid, over the square from -2.5 to 2.5:

```python
lo_v, hi_v = np.array([-2.5, -2.5]), np.array([2.5, 2.5])
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
scatter_codes(Zv_test, axes[0], "VAE: codes (means)")
axes[0].set_xlim(lo_v[0], hi_v[0])
axes[0].set_ylim(lo_v[1], hi_v[1])
axes[1].imshow(decode_grid(vae.decoder, lo_v, hi_v), cmap="gray_r")
axes[1].set_title("decoded grid over the same square")
axes[1].axis("off")
plt.show()
```

Now the codes cover most of the square, and so does the grid: nearly every tile is a plausible garment, and
neighbors morph smoothly into each other -- long-sleeved tops on the upper left turn into T-shirts,
dresses and then trousers toward the upper right, and along the bottom, bags give way to ankle boots and
then sneakers. The few faint tiles sit on the border between footwear and clothing. Same layers, same
data, same optimizer: only the **loss** changed, and it changed what the latent space is good for. Step
8's face sliders depend on exactly this smoothness.

The loss also explains a weakness all of these models share: they **blur**. MSE grades an image pixel by
pixel and knows nothing about what it shows. Shift each test image 2 pixels to the right -- a perfect,
sharp copy to any human -- and compare its MSE with that of Step 1's blurred reconstructions:

```python
x = X_test[:1000]
shifted = torch.roll(x, shifts=2, dims=3)

def edge_strength(imgs):
    """Mean jump between neighboring pixels: a sharpness score."""
    dx = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean()
    dy = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean()
    return (dx + dy).item()

recon = reconstruct(ae, x)
print(f"MSE, shifted 2 px vs original: {mse(shifted, x):.4f}")
print(f"MSE, reconstruction vs original: {mse(recon, x):.4f}")
print(f"edge strength: original {edge_strength(x):.3f}, "
      f"reconstruction {edge_strength(recon):.3f}")
```

~~~text
MSE, shifted 2 px vs original: 0.0750
MSE, reconstruction vs original: 0.0123
edge strength: original 0.158, reconstruction 0.111
~~~

Shifting the garments 2 pixels -- a change no person would call a different image -- costs an MSE of
**0.0750**, six times the **0.0123** of the blurry reconstructions. To MSE a smeared guess is "better"
than a perfect image in the wrong place, so a model trained with it learns to hedge: unsure exactly
where an edge goes, it smears it. The edge strength (the average jump between neighboring pixels)
measures the result: the reconstructions keep 0.111 of the originals' 0.158, about 70%. Step 7 shows a
model trained with a loss that does not reward smearing.

### Step 4: A Denoising Autoencoder

Feed the autoencoder a **noisy** image but ask it for the **clean** one. Copying the input would now
reproduce the noise, so to lower the loss the network must learn what the noise is *not* -- the structure
of clothing. `fit_ae` does it with `noise=0.5`: Gaussian noise with standard deviation 0.5, clipped back
to [0, 1].

```python
dae, _ = fit_ae(lambda: MLPAutoencoder(latent=32), noise=0.5)

torch.manual_seed(123)
noisy = (X_test + 0.5 * torch.randn_like(X_test)).clamp(0, 1)
cleaned = reconstruct(dae, noisy)
print(f"noisy    vs clean, test MSE: {mse(noisy, X_test):.4f}")
print(f"denoised vs clean, test MSE: {mse(cleaned, X_test):.4f}")
show_rows([noisy, cleaned, X_test],
          ["noisy\ninput", "denoised", "clean\noriginal"])
```

~~~text
noisy    vs clean, test MSE: 0.1178
denoised vs clean, test MSE: 0.0187
~~~

The noise pushed the images **0.118** away from clean; the denoiser brought them back to **0.019** --
a sixfold cut, on noise drawn fresh for the test set (seed 123) that the model never saw. The denoised
garments are plainly right even where the noisy input is hard to read, like the boot and the pullover;
like Step 1's reconstructions, they lose the fine print.

### Step 5: A Convolutional Autoencoder

For images, the encoder can be L18's CNN and the decoder its mirror image. Two stride-2 convolutions
shrink 28x28 to 14x14 to 7x7; a dense layer compresses the 32x7x7 feature map to the same 32-number code
as Step 1. The decoder runs the path backwards with **transposed convolutions** (`nn.ConvTranspose2d`),
which *grow* a feature map: with kernel 3, stride 2, padding 1 and `output_padding=1`, each one doubles
the height and width, $(n - 1) \cdot 2 - 2 + 3 + 1 = 2n$.

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),                          # 16 x 14 x 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),                          # 32 x 7 x 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),                          # 16 x 14 x 14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1,
                               output_padding=1),
            nn.Sigmoid())                       # 1 x 28 x 28

    def forward(self, x):
        return self.decoder(self.encoder(x))

cae, _ = fit_ae(lambda: ConvAutoencoder(latent=32))
for name, model in [("MLP ", ae), ("conv", cae)]:
    rec = reconstruct(model, X_test)
    acc = logreg_accuracy(encode(model, X_train),
                          encode(model, X_test))
    print(f"{name} autoencoder: {n_params(model):7,} parameters, "
          f"test MSE {mse(rec, X_test):.4f}, code accuracy {acc:.3f}")
acc_raw = logreg_accuracy(X_train.flatten(1).cpu().numpy(),
                          X_test.flatten(1).cpu().numpy())
print(f"accuracy from all 784 raw pixels: {acc_raw:.3f}")
show_rows([X_test, reconstruct(ae, X_test[:8]),
           reconstruct(cae, X_test[:8])],
          ["original", "MLP", "conv"])
```

~~~text
MLP  autoencoder: 439,728 parameters, test MSE 0.0125, code accuracy 0.814
conv autoencoder: 111,521 parameters, test MSE 0.0090, code accuracy 0.824
accuracy from all 784 raw pixels: 0.826
~~~

The conv autoencoder has a **quarter** of the MLP's parameters (111,521 against 439,728): its
convolutions share their weights across positions, so they shrink the image to 32x7x7 with under 10,000
weights, while the MLP spends about 200,000 on its first layer alone. Yet it reconstructs better: test MSE **0.0090**
against 0.0125, and in the pictures it keeps a ghost of the "Lee" print that the MLP loses. Both 32-number
codes carry most of the class information: logistic regression on them scores 0.81-0.82, close to the
0.826 it reaches on all 784 raw pixels -- a 24-fold compression learned without labels. (The gap between
the two codes' accuracies is within run-to-run noise; don't read a winner into it.)

### Step 6: Anomaly Detection

An autoencoder trained only on **normal** data learns to reconstruct normal things. Show it something it
never saw and the reconstruction fails. Treat bags as the anomaly: train on the nine other classes, then
score every test image by its own reconstruction error.

```python
BAG = CLASSES.index("Bag")
normal = (y_train != BAG).to(device)
ae_normal, _ = fit_ae(lambda: MLPAutoencoder(latent=32),
                      X=X_train[normal])

err = ((reconstruct(ae_normal, X_test) - X_test) ** 2)
err = err.flatten(1).mean(1).cpu().numpy()
is_bag = (y_test == BAG).numpy()
print(f"mean reconstruction error, normal test images: "
      f"{err[~is_bag].mean():.4f}")
print(f"mean reconstruction error, bags (never seen):  "
      f"{err[is_bag].mean():.4f}")
print(f"ROC AUC of 'error' as a bag detector: "
      f"{roc_auc_score(is_bag, err):.3f}")

bins = np.linspace(0, np.percentile(err, 99.5), 60)
plt.figure(figsize=(8, 3.5))
plt.hist(err[~is_bag], bins=bins, alpha=0.6, density=True,
         label="normal (9 classes seen in training)")
plt.hist(err[is_bag], bins=bins, alpha=0.6, density=True,
         label="bags (never seen)")
plt.xlabel("reconstruction error (MSE per image)")
plt.ylabel("density")
plt.legend()
plt.show()
```

~~~text
mean reconstruction error, normal test images: 0.0118
mean reconstruction error, bags (never seen):  0.0328
ROC AUC of 'error' as a bag detector: 0.942
~~~

Bags reconstruct almost three times worse than everything else (0.0328 against 0.0118), and the two
histograms barely overlap: thresholding the error alone detects bags with an ROC AUC of **0.942** (L11:
1.0 is perfect, 0.5 is chance). The model never saw a bag and was never told what one is -- it only
learned what normal looks like. Know the method's limit, too: an anomaly that *looks like* normal data
reconstructs well and hides. Change `"Bag"` to `"Sneaker"` -- sneakers resemble the sandals and ankle
boots the model still sees -- and the AUC drops to about 0.6.

## Part B -- Autoencoders Someone Else Trained

### Step 7: Stable Diffusion's Autoencoder

Stable Diffusion generates images in the latent space of a **convolutional VAE** and lets its decoder
paint the pixels. That autoencoder is public: `stabilityai/sd-vae-ft-mse` on the Hugging Face Hub (MIT
licence, 84 million parameters, a 335 MB download that takes well under a minute). We load it with the
`diffusers` library, as L19 loaded a pretrained ResNet with torchvision, and also load **TAESD**
(`madebyollin/taesd`, MIT), a tiny 2.4-million-parameter autoencoder distilled to imitate it -- the one
to use when speed matters, for example on a CPU.

```python
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.utils import logging as diffusers_logging
from skimage import data

diffusers_logging.set_verbosity_error()   # hide load-time notices
# a deprecation notice from inside huggingface_hub, not our code
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks")
sd_vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    use_safetensors=True).to(device).eval()
tiny_vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", use_safetensors=True).to(device).eval()
print(f"SD VAE: {n_params(sd_vae) / 1e6:.1f} M parameters")
print(f"TAESD:  {n_params(tiny_vae) / 1e6:.1f} M parameters")
```

~~~text
SD VAE: 83.7 M parameters
TAESD:  2.4 M parameters
~~~

Both expect an RGB image as a tensor of shape `[1, 3, H, W]` scaled to $[-1, 1]$, with H and W multiples
of 8. Encode a photo of a cat that ships with scikit-image (CC0), look at the code's shape, and decode
it:

```python
def to_model(img):
    """H x W x 3 uint8 photo -> [1, 3, H, W] tensor in [-1, 1]."""
    h, w = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8
    x = torch.from_numpy(img[:h, :w]).permute(2, 0, 1)
    return (x.float() / 127.5 - 1).unsqueeze(0).to(device)

def to_image(x):
    """[1, 3, H, W] tensor in [-1, 1] -> H x W x 3 in [0, 1]."""
    x = (x.clamp(-1, 1) + 1) / 2
    return x[0].permute(1, 2, 0).cpu().numpy()

photo = to_model(data.chelsea())
with torch.no_grad():
    z = sd_vae.encode(photo).latent_dist.mean   # the mean code
    photo_sd = sd_vae.decode(z).sample
    photo_tiny = tiny_vae.decode(tiny_vae.encode(photo).latents).sample

print("photo shape:", tuple(photo.shape),
      f"= {photo.numel():,} numbers")
print("code shape: ", tuple(z.shape), f"= {z.numel():,} numbers",
      f"({photo.numel() / z.numel():.0f}x fewer)")
for name, out in [("SD VAE", photo_sd), ("TAESD ", photo_tiny)]:
    print(f"{name}: MSE {mse((out + 1) / 2, (photo + 1) / 2):.4f}, "
          f"edges kept {edge_strength(out) / edge_strength(photo):.0%}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, img, title in zip(axes, [photo, photo_sd, photo_tiny],
                          ["original", "SD VAE", "TAESD"]):
    ax.imshow(to_image(img))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

~~~text
photo shape: (1, 3, 296, 448) = 397,824 numbers
code shape:  (1, 4, 37, 56) = 8,288 numbers (48x fewer)
SD VAE: MSE 0.0012, edges kept 87%
TAESD : MSE 0.0019, edges kept 95%
~~~

The photo's 397,824 numbers become a code of 8,288 -- **48 times fewer** -- and the SD VAE gives the
cat back almost exactly: whiskers, irises, fur, an MSE of 0.0012. TAESD, with 35 times fewer
parameters, is close behind (MSE 0.0019). Both keep most of the edge strength -- compare the 70% our
FashionMNIST autoencoder kept in Step 3. (TAESD even scores higher on edges while matching the photo
less closely: edge strength measures sharpness, not faithfulness.) The difference from our models is
not size alone. Stable Diffusion's autoencoder was trained with a **perceptual** loss, which compares
images by the features a pretrained CNN sees in them, and an **adversarial** loss, which penalizes
anything a critic network can tell apart from a real photo (this checkpoint's decoder was then fine-tuned
with MSE plus a perceptual term). Neither rewards smearing.

What does the code look like? It is not a list of numbers like ours -- it is itself a small image, with
4 channels:

```python
fig, axes = plt.subplots(1, 4, figsize=(15, 3))
for c, ax in enumerate(axes):
    ax.imshow(z[0, c].cpu(), cmap="RdBu_r")
    ax.set_title(f"code channel {c}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

Each channel is a 37x56 map that still looks like the cat -- eyes, nose and the outline of the head are
visible -- because the encoder convolved and shrank the photo 8-fold in each direction. This **spatial**
code is what Stable Diffusion's generator works in: it produces a 64x64x4 code from text and noise, and
this decoder paints the 512x512 picture. It also means no single number in the code means anything
like "smile" -- each describes one small patch. For that kind of control we need a vector code, like
Step 8's.

```python
# Your turn (warm-up): Stable Diffusion's own pictures are
# 512 x 512. Predict the shape of the code for one, then check.
# Uncomment and fill the blanks:
# predicted = (1, 4, ____, ____)
# with torch.no_grad():
#     blank = torch.zeros(1, 3, 512, 512, device=device)
#     z512 = sd_vae.encode(blank).latent_dist.mean
# assert tuple(z512.shape) == predicted
# print("code shape:", tuple(z512.shape),
#       f"({3 * 512 * 512 / z512.numel():.0f}x fewer numbers)")
```

The encoder shrinks each side 8-fold (296 to 37 and 448 to 56 above), so a 512x512 picture gets a
64x64 code: `predicted = (1, 4, 64, 64)`, and the check prints `code shape: (1, 4, 64, 64) (48x fewer
numbers)` -- 16,384 numbers for 786,432. That 64x64x4 grid is the space Stable Diffusion generates in.

**Disturb the code.** What does the decoder make of a code that is slightly wrong? Add Gaussian noise to
the cat's code and decode it again. The two models keep their codes on different scales, so the noise is
sized relative to each code's own spread: `level=0.25` means a quarter of the code's typical size.

```python
with torch.no_grad():
    z_tiny = tiny_vae.encode(photo).latents   # TAESD's own code
print(f"code spread (std): SD VAE {z.std():.2f}, "
      f"TAESD {z_tiny.std():.2f}")

def decode_noisy(vae, code, level, seed=0):
    """Add Gaussian noise of size level x (the code's own std)."""
    g = torch.Generator(device=code.device).manual_seed(seed)
    noise = torch.randn(code.shape, generator=g, device=code.device)
    with torch.no_grad():
        return vae.decode(code + level * code.std() * noise).sample

levels = [0, 0.1, 0.25, 0.5, 1.0]
fig, axes = plt.subplots(2, len(levels), figsize=(3.2 * len(levels), 5))
for row, (name, vae, code) in enumerate([("SD VAE", sd_vae, z),
                                         ("TAESD", tiny_vae, z_tiny)]):
    for col, level in enumerate(levels):
        out = decode_noisy(vae, code, level)
        err = mse((out + 1) / 2, (photo + 1) / 2)
        ax = axes[row, col]
        ax.imshow(to_image(out))
        ax.set_title(f"{name}, noise {level}\nMSE {err:.4f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()
```

~~~text
code spread (std): SD VAE 3.78, TAESD 0.65
~~~

Noise of a tenth of the code's spread changes nothing you can see, and a quarter only adds a little
texture to the fur. At 0.5 a painterly texture spreads over the whole picture, and at 1.0 the cat
dissolves into a mosaic of patches -- yet the eyes and the outline of the head are still where they were.
Each number in this spatial code describes one small patch, so the noise changes the texture of each
region, never what the picture shows. (The printed spreads also show that TAESD's code comes about six
times smaller than the SD VAE's: it is pre-scaled to the range Stable Diffusion's generator works in,
which is why the noise is sized to each code.)

**Sample the code.** The SD VAE is a variational autoencoder, so its encoder returns a cloud -- a mean
and a spread for every number -- not a single code (`latent_dist`; above we used `.mean`). Draw a sample
from the cat's cloud and decode it, then decode pure noise: random numbers with the same spread as a
real code, the way Step 3's VAE grid and Step 8's faces are made.

```python
with torch.no_grad():
    post = sd_vae.encode(photo).latent_dist
    from_mean = sd_vae.decode(post.mean).sample
    torch.manual_seed(0)
    from_sample = sd_vae.decode(post.sample()).sample
    noise_code = post.mean.std() * torch.randn_like(post.mean)
    from_noise = sd_vae.decode(noise_code).sample
print(f"spread of the code: {post.mean.std():.2f}; "
      f"sigma of each number: median {post.std.median():.4f}")
print(f"sample vs mean, MSE: {mse(from_sample, from_mean):.1e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, img, title in zip(axes, [from_mean, from_sample, from_noise],
                          ["decode the mean", "decode a sample",
                           "decode random noise"]):
    ax.imshow(to_image(img))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

~~~text
spread of the code: 3.78; sigma of each number: median 0.0003
sample vs mean, MSE: 2.4e-09
~~~

The cat's cloud is tiny: each number's sigma is about 0.0003 against a code spread of 3.78, some ten
thousand times smaller, so a sample decodes to the same cat (MSE 2.4e-09 against the mean's decode). And
random noise decodes to a mosaic of colored texture with no object in it. Both have one cause: this
autoencoder was trained with a KL weight of one millionth (beta = 0.000001 in its training
configuration), the far small end of the concept note's beta dial. The KL term barely acted, so the clouds
shrank to points and the codes never filled a standard-normal cloud: the model behaves like a plain
autoencoder with sharp reconstructions, and a random point is not a valid code. Producing a valid code is
the job of Stable Diffusion's other network, the diffusion model, which turns noise into a meaningful
64x64x4 code step by step before this decoder paints it.

### Step 8: Steering a Face VAE with Sliders

The last model is a convolutional VAE trained for this course on 68,536 faces from **CelebA** (a research
dataset of celebrity photos), at 64x64 pixels with a 128-number code. Training it takes a GPU about ten
minutes, so we trained it once and you download the result: the weights, plus a table with the codes of
19,867 other CelebA faces and their 40 yes/no **attributes** (smiling, eyeglasses, bangs, ...). The
recipe, if you want to reproduce it: the `flwrlabs/celeba` copy of CelebA on the Hugging Face Hub, each
photo center-cropped to 148x148 and shrunk to 64x64, the loss of Step 3 (summed squared error instead of
BCE, plus the KL term, beta = 1), Adam at 1e-3 for 30 epochs. Because they are derived from CelebA, these
two files are covered by CelebA's licence (non-commercial research and education only), not by the
course's CC BY licence. The class is the same kind of model as Step 5, larger:

```python
class FaceVAE(nn.Module):
    """A convolutional VAE for 64x64 RGB faces."""

    def __init__(self, latent=128):
        super().__init__()

        def down(c_in, c_out):   # halve height and width
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
                nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))

        def up(c_in, c_out):     # double height and width
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))

        self.encoder = nn.Sequential(
            down(3, 32), down(32, 64), down(64, 128),
            down(128, 256), nn.Flatten())
        self.to_mu = nn.Linear(256 * 4 * 4, latent)
        self.to_logvar = nn.Linear(256 * 4 * 4, latent)
        self.from_z = nn.Linear(latent, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            up(256, 128), up(128, 64), up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        return self.decoder(self.from_z(z).view(-1, 256, 4, 4))
```

```python
from safetensors import safe_open
from safetensors.torch import load_file

# where the two files live in the course repository
URLS = {
    "l20_face_vae.safetensors":
        "https://raw.githubusercontent.com/devomh/cdat4011-2026/main/u05_deep_learning/models/l20_face_vae.safetensors",
    "l20_face_latents.safetensors":
        "https://raw.githubusercontent.com/devomh/cdat4011-2026/main/u05_deep_learning/models/l20_face_latents.safetensors",
}
Path("models").mkdir(exist_ok=True)
for name, url in URLS.items():
    if not (Path("models") / name).exists():   # Colab: download
        urlretrieve(url, Path("models") / name)

face_vae = FaceVAE()
face_vae.load_state_dict(load_file("models/l20_face_vae.safetensors"))
face_vae = face_vae.to(device).eval()

table = load_file("models/l20_face_latents.safetensors")
codes = table["mu"].float()
attrs = table["attributes"].bool()
with safe_open("models/l20_face_latents.safetensors", "pt") as f:
    ATTR = json.loads(f.metadata()["attribute_names"])
print(f"face VAE: {n_params(face_vae):,} parameters")
print("code table:", tuple(codes.shape), "| attributes:", len(ATTR))
```

~~~text
face VAE: 2,958,659 parameters
code table: (19867, 128) | attributes: 40
~~~

Because it is a VAE, **any** code near the origin should decode to a face. Draw 8 codes from the
standard normal and decode them -- these people do not exist:

```python
def show_faces(z, titles=None, size=1.9):
    with torch.no_grad():
        imgs = face_vae.decode(z.to(device)).cpu()
    fig, axes = plt.subplots(1, len(imgs),
                             figsize=(size * len(imgs), size + 0.3))
    for k, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(imgs[k].permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        if titles:
            ax.set_title(titles[k], fontsize=9)
    plt.show()

torch.manual_seed(7)
z_random = torch.randn(8, 128)
show_faces(z_random)
```

Eight blurry faces -- a VAE trained with a pixel loss, at 64x64 -- but unmistakably faces, with
different ages, hair and lighting, each decoded from 128 random numbers. A plain autoencoder could not do
this (Step 3) -- and, as Step 7 showed, neither can Stable Diffusion's autoencoder on its own.

The face VAE's clouds are also wide. Encode the first random face back into a mean and a spread, and
decode five samples from its cloud:

```python
with torch.no_grad():
    face0 = face_vae.decode(z_random[:1].to(device))
    mu, logvar = face_vae.encode(face0)
    sigma = torch.exp(0.5 * logvar)
print(f"spread of the codes: {codes.std():.2f}; "
      f"sigma of this face's code: median {sigma.median():.2f}")

torch.manual_seed(1)
around = torch.cat([mu] + [mu + sigma * torch.randn_like(mu)
                           for _ in range(5)])
show_faces(around.cpu(), ["mean"] + [f"sample {k}" for k in range(1, 6)])
```

~~~text
spread of the codes: 0.62; sigma of this face's code: median 0.79
~~~

Here the cloud is wide -- the sigma of each number (0.79) is larger than the spread of the codes across
all 19,867 faces in the table (0.62) -- because this model was trained with beta = 1. The five samples
are the same face with different hair (darker or lighter, fuller or flatter), lighting and small changes of
shape: nearby codes mean "a plausible variant of this face". The two VAEs of Part B sit at opposite ends
of the beta dial. A large beta gives a smooth space you can sample and edit, with blurry pictures; a tiny
beta gives sharp pictures and leaves generation to another model.

An **attribute direction** is the difference between the average code of the faces that have an
attribute and the average code of those that do not. Adding it to a code should add the attribute:

```python
def direction(attribute):
    has = attrs[:, ATTR.index(attribute)]
    return codes[has].mean(0) - codes[~has].mean(0)

SLIDERS = ["Smiling", "Eyeglasses", "Bangs", "Blond_Hair"]
DIRS = {a: direction(a) for a in SLIDERS}
for a in SLIDERS:
    share = attrs[:, ATTR.index(a)].float().mean().item()
    print(f"{a:20s} {share:5.1%} of faces, "
          f"direction length {DIRS[a].norm():.2f}")
```

~~~text
Smiling              48.3% of faces, direction length 1.49
Eyeglasses            7.0% of faces, direction length 1.74
Bangs                14.7% of faces, direction length 1.63
Blond_Hair           15.4% of faces, direction length 1.59
~~~

Now the sliders. Pick a face (`face` = which of the 8 random codes above) and move it along each
direction; 0 is the original face, 1 adds one full "average difference":

```python
from ipywidgets import interact

def edit_face(face=0, Smiling=0.0, Eyeglasses=0.0, Bangs=0.0,
              Blond_Hair=0.0):
    amount = dict(Smiling=Smiling, Eyeglasses=Eyeglasses,
                  Bangs=Bangs, Blond_Hair=Blond_Hair)
    z = z_random[face].clone()
    for a, t in amount.items():
        z += t * DIRS[a]
    show_faces(z.unsqueeze(0), size=3.5)

interact(edit_face, face=(0, 7, 1),
         Smiling=(-2, 2, 0.25), Eyeglasses=(-2, 2, 0.25),
         Bangs=(-2, 2, 0.25), Blond_Hair=(-2, 2, 0.25));
```

No slider (or reading this page)? The same edits as fixed strips: face 0, each direction at
-2, -1, 0, +1, +2:

```python
steps = [-2, -1, 0, 1, 2]
for a in SLIDERS:
    z = torch.stack([z_random[0] + t * DIRS[a] for t in steps])
    print(a)
    show_faces(z, [f"{t:+d}" for t in steps], size=1.6)
```

~~~text
Smiling
Eyeglasses
Bangs
Blond_Hair
~~~

Smiling and bangs do what they say, and the blond direction lightens the hair. Eyeglasses adds
glasses -- and more: at +2 the face is also older and grayer and looks more like a man's. The direction is
an average difference between photos of people with and without glasses, and in CelebA the people who
wear glasses are more often older men, so the direction carries all of that together. Exercise 2
measures it.

## Your Turn

### Exercise 1 -- The Code Size Is a Dial

Train Step 1's MLP autoencoder with a code of 2, 4, 8, 16 and 32 numbers. For each, report the test
reconstruction MSE and the logistic-regression accuracy from its code (Step 5's measurement). How do the
two move as the code grows, and do they move together?

**Hint:** loop over the sizes with `fit_ae(lambda: MLPAutoencoder(latent=k))` -- `fit_ae` seeds itself,
so the runs are comparable -- then `mse(reconstruct(...), X_test)` and
`logreg_accuracy(encode(m, X_train), encode(m, X_test))`. On a GPU the five runs take about a minute.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
code  2: test MSE 0.0269, accuracy 0.582
code  4: test MSE 0.0193, accuracy 0.734
code  8: test MSE 0.0152, accuracy 0.765
code 16: test MSE 0.0130, accuracy 0.789
code 32: test MSE 0.0125, accuracy 0.814
~~~

One solution:

~~~python
for k in [2, 4, 8, 16, 32]:
    m, _ = fit_ae(lambda: MLPAutoencoder(latent=k))
    acc = logreg_accuracy(encode(m, X_train), encode(m, X_test))
    print(f"code {k:2d}: test MSE "
          f"{mse(reconstruct(m, X_test), X_test):.4f}, "
          f"accuracy {acc:.3f}")
~~~

The rows for 2, 4 and 32 repeat Steps 2 and 5 exactly, because `fit_ae` re-seeds itself. Both measures
improve as the code grows, with diminishing returns: the error falls by almost half from 2 to 8 numbers and then barely
moves (0.0130 at 16, 0.0125 at 32). But they do not move in lockstep: from 16 to 32 the reconstruction
gains almost nothing, while the accuracy from the code still gains about two points. Reconstruction
error is not a reliable stand-in for how useful a code is -- measure the task you care about.
</details>

### Exercise 2 -- What Else Does a Direction Carry?

The eyeglasses direction was computed from real photos, and in CelebA people who wear glasses are not a
random sample of everyone. Measure what else the direction changes: for each other attribute, the
**cosine similarity** between its direction and the eyeglasses direction (1 = same direction, 0 =
unrelated). Which attributes are most aligned with it? Then move a few random faces along the eyeglasses
direction and look for those changes.

**Hint:** `F.cosine_similarity(direction(a), direction("Eyeglasses"), dim=0)` for every `a` in `ATTR`
except "Eyeglasses"; sort and print the top 5. Then reuse the strip code with `a = "Eyeglasses"` on faces
1-3.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
Chubby           +0.38
Double_Chin      +0.38
Bald             +0.36
Gray_Hair        +0.34
Wearing_Necktie  +0.33
~~~

One solution:

~~~python
d_glasses = direction("Eyeglasses")
sims = {a: F.cosine_similarity(direction(a), d_glasses,
                               dim=0).item()
        for a in ATTR if a != "Eyeglasses"}
for a in sorted(sims, key=sims.get, reverse=True)[:5]:
    print(f"{a:16s} {sims[a]:+.2f}")
~~~

Every attribute most aligned with "eyeglasses" describes an older man: chubby, double chin, bald, gray
hair, a necktie ("Male" is next, at +0.32). The direction was computed from photos, so it learned what
glasses *come with* in this dataset, not glasses alone -- and the strips show it: push faces 1-3 along it
and they age and turn grayer. This is dataset bias, carried straight into the latent space. Getting clean,
separate directions takes more than averaging (balanced data, or models trained to separate the factors),
and the same caution applies to any "attribute direction" you read about.
</details>

## Summary

- An **autoencoder** trains with **no labels**: its target is its own input. A 32-number MLP
  autoencoder reconstructed FashionMNIST at a test MSE of **0.0125**, keeping the shapes and losing the
  fine print.
- The **latent space** organizes itself by similarity: 2 unsupervised numbers classified at about
  **0.6**, 4 at **0.73** -- yet every 2-D projection of the 4-D code scored lower than the dedicated 2-D code.
  Projections are shadows.
- **The loss decides what is learned.** The same network with a **VAE** loss packed its codes around the
  origin and decoded a grid that morphs smoothly from garment to garment; and every pixel loss blurs -- a 2-pixel shift scored six
  times worse than a smeared reconstruction.
- A **denoising** autoencoder cut the noise's MSE sixfold (0.118 to 0.019); a **convolutional** one
  reconstructed better (0.0090) with a quarter of the parameters; and reconstruction error alone flagged
  never-seen bags with an ROC AUC of **0.942**.
- **Stable Diffusion's** autoencoder squeezed a photo 48-fold into a spatial code and gave it back sharp
  -- the payoff of perceptual and adversarial losses. Noise in that code changed texture, not content, and
  with a KL weight of one millionth it cannot be sampled: random codes decode to texture. The face VAE
  (beta = 1) can be sampled, and its code steered along attribute directions, which also carry the
  dataset's biases.
- Next (L21): instead of compressing its whole input into one code, a model with **attention** decides
  which parts of its input to look at -- the idea behind Transformers.
