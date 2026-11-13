---
title: "Lab: Attention from Scratch to Movie Reviews"
unit: "V"
lesson: "21"
type: lab
tags: [deep-learning, pytorch, attention, self-attention, query-key-value, scaled-dot-product, padding-mask, attention-pooling, sentiment-analysis, imdb]
difficulty: intermediate
duration: "90 mins"
---

**Goal:** build attention, take it apart, and test it on real data. Part A works with small tensors:
the **worked example** from the concept page computed stage by stage, an interpretable
**self-attention** matrix, the effect of **learned projections**, the **sqrt(d_k)** scale with its effect on
the gradient, a **lookup** that only attention can solve, and a **reach** study against a CNN. Part B
trains a sentiment classifier on 25,000 **IMDB movie reviews** and compares attention with two cheaper
models, so you see what attention buys and what it costs. Pairs with the concept note
[Attention Mechanisms](l21_concept_attention.qmd).

> **Previously:** L20 -- Autoencoders & Representation Learning (features without labels)  |  **Next:** L22 -- The Transformer Architecture

> This page is the read-only view. To run the lab, open the notebook (`l21_lab_attention.ipynb`) -- in Colab via the badge below, or locally. **Use a GPU runtime** (Runtime > Change runtime type > T4 GPU): Part B trains three text models. Everything also runs on a CPU, a few times slower. The outputs shown on this page came from a CPU, so your numbers on a GPU or another machine may differ slightly.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u05_deep_learning/l21_lab_attention.ipynb)

## Scenario

A movie review says "Great story and great lead actors ... but the movie suffers from bad directing".
Which words decide the verdict, and how far apart are they? Every network so far had fixed wiring: an
MLP mixes a fixed vector, a CNN looks at a small window. **Attention** lets each position decide, from
content, which other positions to read. We build it from one formula, check what it can and cannot do
on small tensors, and then put it to work on real reviews.

## Setup

Two cells: the first installs, the second imports, seeds and picks the device. The IMDB data is
downloaded in Step 6.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships these, so this is effectively a no-op there.
%pip install -q torch matplotlib pandas pyarrow
# local, in a terminal (not in the notebook):
#   uv add torch matplotlib pandas pyarrow
```

```python
# Setup, cell 2 of 2 -- IMPORTS + SEEDS + DEVICE (safe to re-run)
import re
import collections
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# use the GPU when the runtime has one
device = "cuda" if torch.cuda.is_available() else "cpu"
np.set_printoptions(precision=3, suppress=True)
torch.manual_seed(0)
np.random.seed(0)
print("setup ready -", device)
```

~~~text
setup ready - cpu
~~~

## Part A -- Attention on Small Tensors

### Step 1: The Worked Example, Stage by Stage

Attention has three inputs. Every position emits a **query** (what it is looking for), a **key** (what
it offers as a match) and a **value** (what it hands over when matched). The whole layer is one formula:

$$\text{attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

The concept page works this through by hand for three tokens with 2-dimensional queries and keys. Here
is the same example, one stage at a time:

```python
Q = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[10., 0.], [0., 10.], [5., 5.]])
d_k = Q.shape[-1]

scores = Q @ K.T                     # query DOT each key
scaled = scores / d_k ** 0.5         # divide by sqrt(d_k)
weights = scaled.softmax(dim=-1)     # softmax along each row
output = weights @ V                 # blend the values

print("scores Q K^T:\n", scores.numpy())
print("scaled by sqrt(2):\n", scaled.numpy())
print("weights (each row sums to 1):\n", weights.numpy())
print("output:\n", output.numpy())
```

~~~text
scores Q K^T:
 [[1. 0. 1.]
 [0. 2. 2.]
 [1. 1. 2.]]
scaled by sqrt(2):
 [[0.707 0.    0.707]
 [0.    1.414 1.414]
 [0.707 0.707 1.414]]
weights (each row sums to 1):
 [[0.401 0.198 0.401]
 [0.108 0.446 0.446]
 [0.248 0.248 0.503]]
output:
 [[6.017 3.983]
 [3.313 6.687]
 [5.    5.   ]]
~~~

Row 1 is token 1's query `[1, 0]`. It matches key 1 and key 3 equally well (score 1 each) and key 2 not
at all, so the softmax gives keys 1 and 3 the same weight and key 2 less. The output for token 1 is
mostly value 1 and value 3, blended. Now package the four lines as a function and check it against
PyTorch's own implementation:

```python
def attention(Q, K, V):
    """softmax(Q K^T / sqrt(d_k)) V -> (output, weights)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    weights = scores.softmax(dim=-1)
    return weights @ V, weights

out, w = attention(Q, K, V)
ref = F.scaled_dot_product_attention(Q, K, V)
print("our output == PyTorch's (to 1e-6):",
      torch.allclose(out, ref, atol=1e-6))
```

~~~text
our output == PyTorch's (to 1e-6): True
~~~

<details><summary>(Optional) Reading the code: <code>transpose(-2, -1)</code>, batched <code>@</code>, <code>F.scaled_dot_product_attention</code></summary>

**`K.transpose(-2, -1)`** swaps the last two dimensions. For a single sequence `K` is
`(length, d_k)` and this is just `K.T`. Later the tensors get a batch dimension in front,
`(batch, length, d_k)`; `K.T` would then reverse *all* dimensions, while `transpose(-2, -1)` still swaps
only the last two, which is what the formula needs.

**`@` on 3-D tensors** multiplies matrix by matrix for every item in the batch at once:
`(B, L, d) @ (B, d, L)` gives `(B, L, L)`, one score matrix per sequence. That is why the same
`attention` function works for one sequence here and for a batch of 1000 in Step 4.

**`F.scaled_dot_product_attention(Q, K, V)`** is PyTorch's built-in version of the same formula. It
picks a fast implementation for your hardware (on a GPU, a fused kernel), so its last digits can differ
from ours, which is why we compare with `allclose` and a tolerance instead of `==`. It returns only the
output, not the weights.
</details>

### Step 2: Self-Attention and Learned Projections

It is **self**-attention when the queries, keys and values all come from the **same** sequence. The
simplest version uses each token as its own query, key and value (`Q = K = V = X`). Build four tokens
as two near-duplicate pairs and read the attention matrix:

```python
X = torch.tensor([
    [3.0, 0.6, 0.0, 0.0],     # token 0 -- 'cat'
    [3.0, 0.0, 0.0, 0.0],     # token 1 -- 'kitten'
    [0.0, 0.0, 3.0, 0.6],     # token 2 -- 'car'
    [0.0, 0.0, 3.0, 0.0],     # token 3 -- 'truck'
])
ctx, A = attention(X, X, X)   # query, key, value all from X
print("attention matrix (row i = where token i looks):")
print(A.numpy())

labels = ["cat", "kitten", "car", "truck"]
plt.figure(figsize=(4.5, 4))
plt.imshow(A.numpy(), cmap="Blues", vmin=0, vmax=1)
plt.xticks(range(4), labels); plt.yticks(range(4), labels)
plt.xlabel("attended to"); plt.ylabel("query token")
plt.title("self-attention weights"); plt.colorbar()
plt.show()
```

~~~text
attention matrix (row i = where token i looks):
[[0.539 0.451 0.005 0.005]
 [0.495 0.495 0.005 0.005]
 [0.005 0.005 0.539 0.451]
 [0.005 0.005 0.495 0.495]]
~~~

Two clean blocks: `cat` splits its attention between itself and `kitten` and all but ignores the
vehicles; the vehicles do the mirror image. A token attends to the tokens it resembles.

That is also the limitation of `Q = K = X`. The score between tokens *i* and *j* is the dot product
`x_i . x_j`, which is the same as `x_j . x_i`, so the score matrix is **symmetric**, and a token usually
matches itself best. But useful relations are often one-sided: in "the suitcase was too small, so **it**
stayed home", the pronoun *it* should look for a noun, while the noun has no reason to look for the
pronoun. **Learned projections** fix this. Three weight matrices map each token to a separate query, key
and value: `Q = X Wq`, `K = X Wk`, `V = X Wv`. Here is a hand-set `Wq` that turns a pronoun's query into
"I am looking for a noun":

```python
# features: [is_noun, is_pronoun, is_adjective]
T = 3.0 * torch.eye(3)       # 'suitcase', 'it', 'small'
words = ["suitcase", "it", "small"]

S_plain = T @ T.T            # Q = K = X
print("Q = K = X: scores symmetric?",
      torch.allclose(S_plain, S_plain.T))

Wq = torch.eye(3)
Wq[1] = torch.tensor([1.0, 0.0, 0.0])   # pronoun -> seek noun
Wk = torch.eye(3)
_, A_plain = attention(T, T, T)
_, A_proj = attention(T @ Wq, T @ Wk, T)
for name, A_ in [("Q = K = X", A_plain),
                 ("with Wq", A_proj)]:
    for i in [1, 0]:          # 'it', then 'suitcase'
        top = A_[i].argmax().item()
        print(f"{name:10s}: '{words[i]}' attends most to"
              f" '{words[top]}' ({A_[i, top].item():.3f})")
```

~~~text
Q = K = X: scores symmetric? True
Q = K = X : 'it' attends most to 'it' (0.989)
Q = K = X : 'suitcase' attends most to 'suitcase' (0.989)
with Wq   : 'it' attends most to 'suitcase' (0.989)
with Wq   : 'suitcase' attends most to 'suitcase' (0.989)
~~~

With the raw tokens, *it* attends to itself. With the projection, its query points at the noun
dimension and it attends to *suitcase*, while *suitcase* still attends to itself: the relation is now
one-sided. In a real model nobody sets `Wq`, `Wk` and `Wv` by hand; they are `nn.Linear` layers learned
by backpropagation, as in Step 4.

### Step 3: Why Divide by sqrt(d_k)

A dot product of two `d_k`-dimensional vectors is a sum of `d_k` terms, so its spread grows like
`sqrt(d_k)`. Large scores make the softmax **saturate**: nearly all the weight lands on one key. A
saturated softmax also passes almost no gradient back, because its output barely moves when its inputs
move. The table measures both, over 2000 random queries against 8 keys: the largest attention weight,
and the size of the softmax's Jacobian (how much the weights change per unit change of the scores; 0
means the layer cannot learn through it).

```python
def jacobian_norm(p):
    """Frobenius norm of d softmax / d scores, per row."""
    J = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    return J.flatten(-2).norm(dim=-1)

torch.manual_seed(0)
print(f"{'d_k':>5} {'max weight':>11} {'':>7}"
      f" {'gradient size':>14}")
print(f"{'':>5} {'unscaled':>11} {'scaled':>7}"
      f" {'unscaled':>8} {'scaled':>6}")
for d_k in [4, 16, 64, 256]:
    q = torch.randn(2000, 1, d_k)
    Kk = torch.randn(2000, 8, d_k)
    raw = (q @ Kk.transpose(-2, -1)).squeeze(1)
    p_raw = raw.softmax(-1)
    p_sc = (raw / d_k ** 0.5).softmax(-1)
    print(f"{d_k:>5}"
          f" {p_raw.max(-1).values.mean().item():>11.3f}"
          f" {p_sc.max(-1).values.mean().item():>7.3f}"
          f" {jacobian_norm(p_raw).mean().item():>8.3f}"
          f" {jacobian_norm(p_sc).mean().item():>6.3f}")
```

~~~text
  d_k  max weight          gradient size
         unscaled  scaled unscaled scaled
    4       0.529   0.344    0.340  0.360
   16       0.748   0.359    0.252  0.363
   64       0.875   0.361    0.151  0.366
  256       0.938   0.362    0.082  0.365
~~~

Unscaled, the top weight climbs from 0.53 toward 0.94 as `d_k` grows, and the gradient size falls from
0.340 to 0.082: by 256 dimensions the softmax has nearly collapsed onto one key and passes back a
quarter of the gradient it passed at `d_k = 4`. Scaled, both stay steady near 0.36 at every dimension.
This is L17's vanishing-gradient problem, avoided inside a single layer by one division.

### Step 4: A Content Lookup Only Attention Can Solve

Now let attention learn its own `Wq`, `Wk` and `Wv`. Each sequence is a **query** token (position 0)
followed by seven **memory slots**; every slot has a distinct **key** and a random **value**. The
query's key matches exactly one slot, and the answer is that slot's value. Solving it means *finding
the slot whose key matches*, wherever it sits: a lookup by content.

```python
def make_recall_dataset(n, seq_len=8, n_keys=10, n_vals=6,
                        seed=0):
    """Position 0 is a QUERY whose key matches one memory
    slot; positions 1.. are slots with distinct keys and
    random values. Target = the matched slot's value."""
    g = torch.Generator().manual_seed(seed)
    n_slots = seq_len - 1
    PAD_VAL = n_vals
    keys = torch.zeros(n, seq_len, dtype=torch.long)
    vals = torch.full((n, seq_len), PAD_VAL, dtype=torch.long)
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        slot_keys = torch.randperm(n_keys, generator=g)[:n_slots]
        slot_vals = torch.randint(n_vals, (n_slots,), generator=g)
        tgt = torch.randint(n_slots, (1,), generator=g).item()
        keys[i, 1:] = slot_keys
        vals[i, 1:] = slot_vals
        keys[i, 0] = slot_keys[tgt]   # query asks for this key
        y[i] = slot_vals[tgt]         # answer = its value
    return keys, vals, y

ktr, vtr, ytr = make_recall_dataset(4000, seed=0)
kte, vte, yte = make_recall_dataset(1000, seed=1)
data = (ktr, vtr, ytr, kte, vte, yte)
print("train", tuple(ktr.shape), "test", tuple(kte.shape))
print("example 0 -> query key:", kte[0, 0].item(),
      "| slot keys:", kte[0, 1:].tolist())
print("            slot values:", vte[0, 1:].tolist(),
      "| answer:", yte[0].item())
```

~~~text
train (4000, 8) test (1000, 8)
example 0 -> query key: 9 | slot keys: [5, 6, 1, 2, 0, 8, 9]
            slot values: [0, 2, 3, 4, 5, 1, 0] | answer: 0
~~~

Each discrete key or value symbol becomes a learned vector with `nn.Embedding` (a **token embedding**:
a lookup table the network trains). The attention model projects every token to a query, key and
value, attends, and reads out at the query position. The **baseline** sees the same tokens but
**mean-pools** them, averaging the positions away, so it cannot tell which value belongs to the
query's key.

```python
class SelfAttnRecall(nn.Module):
    def __init__(self, n_keys, n_vals, d=32):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d)
        # +1 for the query's PAD value
        self.val_emb = nn.Embedding(n_vals + 1, d)
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.out = nn.Linear(d, n_vals)

    def forward(self, keys, vals, return_attn=False):
        X = self.key_emb(keys) + self.val_emb(vals)  # (B, L, d)
        ctx, w = attention(self.Wq(X), self.Wk(X), self.Wv(X))
        logits = self.out(ctx[:, 0])  # read out at the query
        return (logits, w) if return_attn else logits

class MeanPoolBaseline(nn.Module):
    def __init__(self, n_keys, n_vals, d=32):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d)
        self.val_emb = nn.Embedding(n_vals + 1, d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                                 nn.Linear(d, n_vals))

    def forward(self, keys, vals):
        X = self.key_emb(keys) + self.val_emb(vals)
        return self.mlp(X.mean(dim=1))  # pools -> cannot route

def train_model(model, data, epochs=20, bs=256):
    """Adam + cross-entropy; returns test accuracy."""
    ktr, vtr, ytr, kte, vte, yte = [t.to(device) for t in data]
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n = len(ytr)
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(epoch)
        perm = torch.randperm(n, generator=g).to(device)
        model.train()
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss_fn(model(ktr[b], vtr[b]), ytr[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(kte, vte).argmax(1)
        return (pred == yte).float().mean().item()

torch.manual_seed(0)
attn_model = SelfAttnRecall(n_keys=10, n_vals=6)
acc_attn = train_model(attn_model, data)

torch.manual_seed(0)
base_model = MeanPoolBaseline(n_keys=10, n_vals=6)
acc_base = train_model(base_model, data)

print(f"self-attention test accuracy: {acc_attn:.4f}")
print(f"mean-pool baseline accuracy:  {acc_base:.4f}"
      f"   (chance = {1/6:.4f})")
```

~~~text
self-attention test accuracy: 1.0000
mean-pool baseline accuracy:  0.3730   (chance = 0.1667)
~~~

Attention solves it **perfectly**. The baseline, with the same information but no way to route it, gets
**0.3730**: above chance (it can lean on which values are present) but far from binding the query to the
right slot. Look at where the trained attention points for one test sequence:

```python
with torch.no_grad():
    _, w = attn_model(kte[:1].to(device), vte[:1].to(device),
                      return_attn=True)
qrow = w[0, 0].cpu().numpy()  # the query's weights
# the slot whose key matches the query
match = (kte[0, 1:] == kte[0, 0]).nonzero().item() + 1
print("query-position attention weights:", qrow)
print("matching slot is position:", match,
      "| attention's argmax:", int(qrow.argmax()))

plt.figure(figsize=(6, 3))
plt.bar(range(len(qrow)), qrow)
plt.axvline(match, color="red", ls="--",
            label=f"matching slot ({match})")
plt.xlabel("position"); plt.ylabel("attention weight")
plt.legend(); plt.title("the query attends to the matching slot")
plt.show()
```

~~~text
query-position attention weights: [0.    0.016 0.005 0.    0.001 0.001 0.014 0.963]
matching slot is position: 7 | attention's argmax: 7
~~~

The query spends **0.963** of its attention on position 7, the one slot whose key matches, and almost
nothing elsewhere. Nobody wrote that rule: the model learned the lookup from 4000 examples.

```python
# Your turn (warm-up): the attention weights for a batch are
# (batch, seq_len, seq_len) -- one L x L matrix per example.
# Predict the shape for ALL 1000 test sequences, then check.
# Uncomment and fill the blank:
# with torch.no_grad():
#     _, W = attn_model(kte.to(device), vte.to(device),
#                       return_attn=True)
# print("attention weights shape:", tuple(W.shape))
# expect (____, ____, ____)
```

The shape is `(1000, 8, 8)`: one 8x8 attention matrix for each of the 1000 test sequences (each of the 8
positions attends over all 8). That `seq_len x seq_len` matrix is why long sequences are expensive -- it
grows with the **square** of the length (Step 7 measures it).

<details><summary>(Optional) Reading the code: <code>nn.Embedding</code></summary>

**`nn.Embedding(num, d)`** is a trainable table with `num` rows of length `d`. Calling it on a tensor
of integer ids returns the matching rows: `key_emb(keys)` turns a `(B, L)` tensor of key ids into a
`(B, L, d)` tensor of vectors. It is equivalent to one-hot encoding each id and multiplying by a weight
matrix (a `nn.Linear` without bias), just without building the one-hot vectors. The rows start random
and are trained by backpropagation like any other weight. Adding two embeddings,
`key_emb(keys) + val_emb(vals)`, gives each position one vector that carries both its key and its
value. In Step 6, `padding_idx=0` tells the table that id 0 is padding: that row stays at zero and is
never updated.
</details>

### Step 5: Reach -- Attention vs a CNN

The concept page claims that attention connects any two positions in one step, while a convolution only
sees a small window, so a CNN needs more layers to see further (L18's receptive field). Test it. In this
task each sequence has 32 positions: position 0 is the query, and exactly one of the other 31 positions
carries a **marker**. The answer is the value stored at the marked position, which can be anywhere
from 1 to 31 positions away. All three models read out at position 0, so each can use only what reaches
position 0.

```python
def make_marker_dataset(n, seq_len=32, n_vals=6, seed=0):
    """vals: random values at positions 1..; marks: 1 at the
    one marked position. Target = the marked value.
    dist = how far the marked position is from position 0."""
    g = torch.Generator().manual_seed(seed)
    marks = torch.zeros(n, seq_len, dtype=torch.long)
    vals = torch.full((n, seq_len), n_vals, dtype=torch.long)
    y = torch.zeros(n, dtype=torch.long)
    dist = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        v = torch.randint(n_vals, (seq_len - 1,), generator=g)
        t = torch.randint(seq_len - 1, (1,), generator=g).item()
        vals[i, 1:] = v
        marks[i, 1 + t] = 1
        y[i] = v[t]
        dist[i] = t + 1
    return marks, vals, y, dist

mtr, mvtr, mytr, _ = make_marker_dataset(8000, seed=0)
mte, mvte, myte, mdist = make_marker_dataset(2000, seed=1)
mdata = (mtr, mvtr, mytr, mte, mvte, myte)

class ConvReader(nn.Module):
    """Stacked 1-D convolutions (kernel 3), read out at
    position 0: each layer extends the reach by one."""
    def __init__(self, n_layers, n_vals=6, d=32):
        super().__init__()
        self.key_emb = nn.Embedding(2, d)
        self.val_emb = nn.Embedding(n_vals + 1, d)
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv1d(d, d, 3, padding=1), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_vals)

    def forward(self, marks, vals):
        X = self.key_emb(marks) + self.val_emb(vals)
        H = self.conv(X.transpose(1, 2))   # (B, d, L)
        return self.out(H[:, :, 0])

def accuracy_by_distance(model, mdata, mdist):
    _, _, _, mte, mvte, myte = mdata
    model.eval()
    with torch.no_grad():
        pred = model(mte.to(device), mvte.to(device)).cpu()
    ok = (pred.argmax(1) == myte).float()
    return [ok[mdist == d].mean().item() for d in range(1, 32)]

curves = {}
torch.manual_seed(0)
m = SelfAttnRecall(n_keys=2, n_vals=6)
train_model(m, mdata, epochs=30)
curves["attention (1 layer)"] = accuracy_by_distance(m, mdata, mdist)
for n_layers in [3, 6]:
    torch.manual_seed(0)
    m = ConvReader(n_layers)
    train_model(m, mdata, epochs=30)
    curves[f"CNN, {n_layers} layers"] = accuracy_by_distance(
        m, mdata, mdist)

print("accuracy by distance   1-3   4-6  7-31")
for name, c in curves.items():
    print(f"{name:20s} {np.mean(c[0:3]):5.2f} {np.mean(c[3:6]):5.2f}"
          f" {np.mean(c[6:]):5.2f}")

plt.figure(figsize=(7, 3.2))
for name, c in curves.items():
    plt.plot(range(1, 32), c, marker="o", ms=3, label=name)
plt.axhline(1 / 6, color="gray", ls=":", label="chance (1/6)")
plt.xlabel("distance from the query to the marked position")
plt.ylabel("test accuracy"); plt.ylim(0, 1.05)
plt.legend(); plt.title("how far can each model reach?")
plt.show()
```

~~~text
accuracy by distance   1-3   4-6  7-31
attention (1 layer)   1.00  1.00  1.00
CNN, 3 layers         1.00  0.17  0.18
CNN, 6 layers         1.00  1.00  0.16
~~~

Each model solves exactly what it can reach. One attention layer scores 1.00 at every distance from 1
to 31. The 3-layer CNN scores 1.00 up to distance 3 and chance (about 1/6) beyond; the 6-layer CNN scores
about 1.00 up to distance 6 and chance beyond. The CNNs are not too weak for the task: inside their reach
they are perfect. They cannot see further, and reaching position 31 would take 31 such layers. (On
another machine the chance-level numbers move by a few hundredths; the steps do not.)

<details><summary>(Optional) Reading the code: <code>nn.Conv1d</code> and the reach of a stack</summary>

**`nn.Conv1d(d, d, 3, padding=1)`** is L18's `nn.Conv2d` with one spatial dimension instead of two. Its
input is `(batch, channels, length)`, which is why the model transposes the `(B, L, d)` embeddings to
`(B, d, L)` first: here the `d` embedding numbers play the role of the image's color channels. Kernel 3
means each output position combines its own position and one neighbor on each side; `padding=1` keeps
the length at 32.

**Reach.** One layer lets position 0 see positions 0 and 1. A second layer sees the first layer's
outputs at positions 0 and 1, each of which saw one step further, so it reaches position 2. Each layer
adds one, so 3 layers reach position 3 and 6 layers reach position 6. To reach position 31 with kernel 3
you would need 31 layers (wider kernels and pooling shorten that, but the reach always grows with
depth). Attention's reach is the whole sequence in one layer.

The attention model is Step 4's `SelfAttnRecall` reused: `n_keys=2` because the "key" input is now
just the marker flag (0 or 1).
</details>

## Part B -- A Real Application: IMDB Movie Reviews

### Step 6: Train Three Sentiment Classifiers

The [IMDB review dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (Maas et al., 2011) holds
50,000 movie reviews labeled positive or negative, split 25,000 / 25,000 into train and test. The
download is two Parquet files of about 21 MB each from the Hugging Face Hub, pinned to a fixed revision
so the data cannot change under the lab. The dataset card lists its licence as "other" without
further terms; cite Maas et al. (2011) if you use the data, and treat the reviews (written by IMDb
users) as material for study, not for redistribution.

```python
REV = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
BASE = ("https://huggingface.co/datasets/stanfordnlp/imdb/"
        f"resolve/{REV}/plain_text/")
imdb_dir = Path("data/imdb")
imdb_dir.mkdir(parents=True, exist_ok=True)
for split in ["train", "test"]:
    f = imdb_dir / f"{split}.parquet"
    if not f.exists():
        urlretrieve(BASE + f"{split}-00000-of-00001.parquet", f)

train_df = pd.read_parquet(imdb_dir / "train.parquet")
test_df = pd.read_parquet(imdb_dir / "test.parquet")
print("train", train_df.shape, "| test", test_df.shape)
print("labels (0 = negative, 1 = positive):",
      train_df["label"].value_counts().sort_index().tolist())
print(test_df["text"][958][:72], "...")
```

~~~text
train (25000, 2) | test (25000, 2)
labels (0 = negative, 1 = positive): [12500, 12500]
Great story and great lead actors (Quaid and Ryan) but the movie suffers ...
~~~

A network cannot read text; it reads integers. **Tokenization** splits each review into words; a
**vocabulary** gives each of the 20,000 most frequent training words an id. Two ids are reserved: 0 is
`<pad>`, which fills short reviews up to a common length, and 1 is `<unk>`, for any word outside the
vocabulary. Reviews are cut at 256 words (the median review has about 170).

```python
VOCAB, MAX_LEN = 20000, 256

def tokenize(text):
    text = text.lower().replace("<br />", " ")
    return re.findall(r"[a-z']+", text)

counts = collections.Counter(
    w for t in train_df["text"] for w in tokenize(t))
itos = ["<pad>", "<unk>"] + [
    w for w, _ in counts.most_common(VOCAB - 2)]
stoi = {w: i for i, w in enumerate(itos)}

def encode(texts, max_len=MAX_LEN):
    """Texts -> (n, max_len) word ids, 0-padded."""
    X = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in tokenize(t)][:max_len]
        X[i, :len(ids)] = torch.tensor(ids)
    return X

# shuffled; 10,000 test reviews keep evaluation fast
test_sub = test_df.sample(frac=1, random_state=1).iloc[:10000]
train_sh = train_df.sample(frac=1, random_state=0)
Xtr_txt = encode(train_sh["text"])
ytr_txt = torch.tensor(train_sh["label"].values)
Xte_txt = encode(test_sub["text"])
yte_txt = torch.tensor(test_sub["label"].values)
print("Xtr_txt", tuple(Xtr_txt.shape), "| Xte_txt", tuple(Xte_txt.shape))
print("first 8 ids of review 0:", Xtr_txt[0, :8].tolist())
print("decoded:", [itos[i] for i in Xtr_txt[0, :8].tolist()])
print(f"reviews longer than {MAX_LEN} words (cut): "
      f"{(Xtr_txt[:, -1] != 0).float().mean().item():.0%}")
```

~~~text
Xtr_txt (25000, 256) | Xte_txt (10000, 256)
first 8 ids of review 0: [16995, 1, 15848, 7, 14, 49, 4, 10241]
decoded: ['fr', '<unk>', 'doktor', 'is', 'as', 'good', 'a', 'demonstration']
reviews longer than 256 words (cut): 29%
~~~

Three models, each with its own 64-dimensional word embedding, so their sizes are almost equal
(the 20,000 x 64 embedding table is most of each):

- **Bag of embeddings:** average the word vectors, then one linear layer. Word order and context are
  ignored. This is the neural cousin of L07's bag of words.
- **1-D CNN:** one convolution over windows of 5 words, then the maximum over positions. It sees local
  word order ("not good") but nothing beyond 5 words.
- **Self-attention + attention pooling:** every word attends to every other word (with a residual
  connection, as in L19), then a single learned query attends over the result to produce one summary
  vector. Padded positions are **masked** so that they get zero weight.

The mask is one new line in `attention`: before the softmax, set the score of every padded key to
minus infinity, so its weight comes out exactly 0. The new version replaces Step 1's and gives the same
result when `mask` is left out. (If you later re-run Step 1's cell, run this one again too: the IMDB
model passes a mask, and Step 1's version does not accept one.)

```python
def attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V; mask (B, L_k) is True
    for real tokens, False for padding."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, :],
                                    float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ V, weights

D = 64

class BagOfEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D, padding_idx=0)
        self.out = nn.Linear(D, 2)

    def forward(self, x):
        m = (x != 0).float().unsqueeze(-1)   # 1 = real word
        mean = (self.emb(x) * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(mean)

class CNN1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D, padding_idx=0)
        self.conv = nn.Conv1d(D, D, 5, padding=2)
        self.out = nn.Linear(D, 2)

    def forward(self, x):
        h = F.relu(self.conv(self.emb(x).transpose(1, 2)))
        h = h.masked_fill((x == 0).unsqueeze(1), -1e9)
        return self.out(h.max(dim=-1).values)

class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D, padding_idx=0)
        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        # the pooling query: one learned vector
        self.pool_q = nn.Parameter(0.1 * torch.randn(1, 1, D))
        self.out = nn.Linear(D, 2)

    def forward(self, x, use_mask=True, return_attn=False):
        mask = (x != 0) if use_mask else None
        E = self.emb(x)
        H, _ = attention(self.Wq(E), self.Wk(E), self.Wv(E),
                         mask)
        H = E + H                     # residual connection
        q = self.pool_q.expand(len(x), -1, -1)
        c, p = attention(q, H, H, mask)   # attention pooling
        logits = self.out(c[:, 0])
        return (logits, p[:, 0]) if return_attn else logits

def fit_text(model, epochs=5, bs=64):
    """Adam + cross-entropy on (Xtr_txt, ytr_txt)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(epoch)
        perm = torch.randperm(len(ytr_txt), generator=g)
        model.train()
        for s in range(0, len(ytr_txt), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            logits = model(Xtr_txt[b].to(device))
            loss_fn(logits, ytr_txt[b].to(device)).backward()
            opt.step()
    return model

@torch.no_grad()
def predict_proba(model, X, bs=500, **kw):
    """P(positive) for every row of X."""
    model.eval()
    out = [model(X[s:s + bs].to(device), **kw).softmax(-1)[:, 1]
           for s in range(0, len(X), bs)]
    return torch.cat(out).cpu()

models = {}
for name, cls in [("bag of embeddings", BagOfEmbeddings),
                  ("1-D CNN", CNN1d),
                  ("self-attention", AttentionClassifier)]:
    torch.manual_seed(0)
    models[name] = fit_text(cls())

print(f"{'model':18s} {'params':>9} {'test acc':>9}")
for name, model in models.items():
    n_par = sum(p.numel() for p in model.parameters())
    acc = ((predict_proba(model, Xte_txt) > 0.5).long() == yte_txt)
    print(f"{name:18s} {n_par:>9,} {acc.float().mean().item():>9.4f}")
```

~~~text
model                 params  test acc
bag of embeddings  1,280,130    0.8662
1-D CNN            1,300,674    0.8395
self-attention     1,292,674    0.8514
~~~

All three are far above chance (0.5), and the **cheapest model wins**: the bag of embeddings scores
0.8662, attention 0.8514 and the CNN 0.8395 (the CNN's last digits vary a little between machines; the
order does not). Movie sentiment is carried mostly by *which* words appear, so averaging the words
already captures most of the signal, and routing between words adds nothing here. It does cost: on the
CPU that produced this page, training took about 5 s for the bag, 19 s for the CNN and 53 s for
attention (on a Colab T4 GPU, about 24 s for all three together). Attention pays off when the answer depends on relations between positions, as in
Steps 4 and 5.

<details><summary>(Optional) Reading the code: <code>masked_fill</code>, <code>nn.Parameter</code>, <code>pd.read_parquet</code></summary>

**`scores.masked_fill(~mask[:, None, :], float("-inf"))`.** `mask` is `(B, L)`, True for real words.
`~mask` flips it to True for padding, and `[:, None, :]` adds a middle dimension so it lines up with
the `(B, L_q, L_k)` scores: the same key mask applies to every query row. `masked_fill` writes minus
infinity wherever the mask is True; `exp(-inf) = 0`, so after the softmax those keys get weight exactly
0 and the remaining weights still sum to 1. The CNN uses the same idea with `-1e9` before its max: a
padded position can never be the maximum.

**`nn.Parameter(0.1 * torch.randn(1, 1, D))`** registers a tensor as a trainable weight of the module,
so `model.parameters()` includes it and Adam updates it. Here it is the pooling query: one vector,
shared by all reviews, that learns *what to look for*. `expand(len(x), -1, -1)` repeats it (without
copying) once per review in the batch.

**`pd.read_parquet`** reads a Parquet file, a compressed column-oriented table format, into a
DataFrame. It needs the `pyarrow` package, which Colab ships.
</details>

### Step 7: What Attention Buys, and What It Costs

**Where does the pooling look?** The pooling query's weights say how much each word position
contributes to the summary vector. Here they are for the review from the scenario:

```python
att_model = models["self-attention"]
hook = test_df["text"][958]
x_hook = encode([hook])
n_words = int((x_hook != 0).sum())
with torch.no_grad():
    att_model.eval()
    logits, p = att_model(x_hook.to(device), return_attn=True)
p = p[0, :n_words].cpu()
words_hook = [itos[i] for i in x_hook[0, :n_words].tolist()]
print(f"true label: {test_df['label'][958]} (negative) | "
      f"P(positive) = {logits.softmax(-1)[0, 1].item():.3f}")
top = torch.argsort(-p.round(decimals=4), stable=True)[:8]
print("most-attended words (even spread would be "
      f"{1 / n_words:.3f} each):")
for i in top.tolist():
    print(f"  {words_hook[i]:12s} {p[i].item():.3f}")

plt.figure(figsize=(10, 2.6))
plt.bar(range(n_words), p.numpy())
plt.xticks(range(n_words), words_hook, rotation=90, fontsize=7)
plt.ylabel("pooling weight")
plt.title("attention pooling over one review")
plt.tight_layout(); plt.show()
```

~~~text
true label: 0 (negative) | P(positive) = 0.035
most-attended words (even spread would be 0.015 each):
  script       0.028
  looked       0.025
  ok           0.024
  save         0.024
  screenplay   0.023
  suffers      0.022
  in           0.021
  look         0.021
~~~

The model is confident the review is negative (P(positive) = 0.035), which is correct. But the
pooling weights are spread thin: the largest is 0.028, while an even spread over this review would give
about 0.015 to each word. Some top words make sense (*script*, *save*, *screenplay*, *suffers*), but
*in* and *looked* are on the list, and none of the four copies of *bad* is. After self-attention, each
position carries a blend of its context, so a weight on a position is not a weight on that word alone.
Exercise 3 tests how far the weights can be trusted.

**Order.** Attention has no notion of position: each word interacts with the others by content only.
Two sentences made of exactly the same words, with opposite meanings, show what that costs:

```python
pairs = ["this movie is not bad , it is good",
         "this movie is not good , it is bad"]
print(f"{'model':18s} {'sentence 1':>10} {'sentence 2':>10}")
for name, model in models.items():
    p1, p2 = predict_proba(model, encode(pairs)).tolist()
    print(f"{name:18s} {p1:>10.4f} {p2:>10.4f}")
```

~~~text
model              sentence 1 sentence 2
bag of embeddings      0.0003     0.0003
1-D CNN                0.0473     0.0781
self-attention         0.8448     0.8448
~~~

The bag of embeddings and attention give the two sentences **identical** probabilities (0.0003 and
0.8448): the sentences contain the same words, and neither model knows their order. Attention even calls
both of them positive. The CNN, which reads windows of 5 words, scores them differently (its exact
numbers vary a little between machines). This is the order-blindness from the concept page, in trained
models; the Transformer (L22) fixes it with positional encoding.

**Cost.** The attention matrix has one entry per pair of positions, so for a sequence of length `L` it
holds `L x L` scores. Measure the time of one attention call, and the memory of its score matrix, as the
length doubles:

```python
import time

rows = []
for L in [256, 512, 1024, 2048, 4096]:
    Xc = torch.randn(1, L, D, device=device)
    attention(Xc, Xc, Xc)               # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        attention(Xc, Xc, Xc)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1000
    rows.append((L, ms))
    print(f"L = {L:5d}: score matrix {L * L * 4 / 2**20:7.2f} MB")

Ls, times = zip(*rows)
plt.figure(figsize=(5, 3.2))
plt.loglog(Ls, times, "o-", base=2, label="measured")
plt.loglog(Ls, [times[0] * (L / Ls[0]) ** 2 for L in Ls],
           "--", base=2, label="grows like L^2")
plt.xlabel("sequence length L"); plt.ylabel("ms per call")
plt.legend(); plt.title(f"attention cost on {device}")
plt.show()
```

~~~text
L =   256: score matrix    0.25 MB
L =   512: score matrix    1.00 MB
L =  1024: score matrix    4.00 MB
L =  2048: score matrix   16.00 MB
L =  4096: score matrix   64.00 MB
~~~

The memory column is exact: the score matrix holds `L x L` numbers of 4 bytes each, so doubling `L`
quadruples it, and one 4,096-token sequence already needs 64 MB in a single attention layer. The time plot
is measured on your hardware, so it differs from the one shown here: for short sequences fixed overheads
dominate (on a GPU the first points are nearly flat), and for long ones the time follows the `L^2` line.
This quadratic cost is the main limit on the context length of language models.

## Your Turn

### Exercise 1 -- Equivariant inside, invariant at the readout

Take the first test sequence of Step 4's lookup and shuffle its seven memory slots (positions 1..7),
keeping the query at position 0. Then check two things on the trained `attn_model`: (a) are the
**logits** the same as before (to 1e-6)? (b) inside the layer, are the self-attention outputs
`ctx` of the shuffled sequence the original outputs **shuffled the same way**?

**Hint:** `perm = torch.cat([torch.tensor([0]), torch.randperm(7, generator=torch.Generator().manual_seed(7)) + 1])`
reorders positions with the query fixed. For (b), compute
`X = attn_model.key_emb(k) + attn_model.val_emb(v)` for both versions, run
`attention(attn_model.Wq(X), attn_model.Wk(X), attn_model.Wv(X))`, and compare `ctx_shuffled` with
`ctx_original[:, perm]` using `torch.allclose(..., atol=1e-6)`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
slot keys before: [5, 6, 1, 2, 0, 8, 9]
slot keys after:  [8, 5, 2, 6, 1, 9, 0]
(a) logits unchanged: True
(b) outputs shuffled the same way: True
    outputs unchanged in place:    False
~~~

**(a)** The logits are unchanged: the readout at the query position is a weighted sum over the slots,
and a sum does not care about order, so the prediction is **invariant** to the shuffle. **(b)** Inside the
layer each output vector is also unchanged, but it moves with its token: the shuffled sequence's outputs
equal the original outputs reordered by the same permutation (**equivariant**), and they do not match
position by position (the last line). For this lookup, ignoring order is exactly right. For language,
where order is meaning, the model has to be told the positions: **positional encoding**, in L22.
</details>

### Exercise 2 -- Forget the mask

Take the first 200 rows of `Xte_txt` that are shorter than 100 words. Score them twice with the trained attention
classifier: once padded to 256 positions as usual, and once with 256 more padding positions appended
(512 in total). Do it with the mask on (`use_mask=True`) and off (`use_mask=False`). How much does
P(positive) move, and do any predicted labels flip?

**Hint:** `short = Xte_txt[(Xte_txt != 0).sum(1) < 100][:200]`; `longer = torch.cat([short, torch.zeros_like(short)], 1)`;
`predict_proba(att_model, short, use_mask=...)` passes the flag through to the model.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
mask on : mean change 0.0000, max 0.0000, labels flipped 0/200
mask off: mean change 0.1140, max 0.2058, labels flipped 2/200
~~~

With the mask, the extra padding changes nothing (0.0000). Without it, every word also attends to the
padding positions, and the predictions drift: P(positive) moves by 0.11 on average and by up to 0.21, and
2 of the 200 labels flip. (Retrained with other seeds, the flips ranged from 2 to 57 in our tests.) No
error is raised: the model simply gives answers that depend on how much padding the batch needed. That
is why every attention layer that sees padded batches takes a mask.
</details>

### Exercise 3 -- Is attention an explanation?

If the pooling weights *explain* the prediction, then the word with the highest weight should be the
word the prediction depends on most. Test it on the first 200 rows of `Xte_txt`: for each review, delete
each word in turn (shift the rest left, pad the end), record how much P(positive) moves, and find the
rank of the most-attended word among all the words by that change (rank 1 = the most important word).

**Hint:** for one review `x` (shape `(1, 256)`) with `n` words, build all `n` deletions at once:
`Y = x.repeat(n, 1)` and, for each `j`, `Y[j] = torch.cat([x[0, :j], x[0, j+1:], x.new_zeros(1)])`. The
change is `(predict_proba(att_model, Y) - p0).abs()`. Count the rank as 1 + the number of words whose
change is strictly larger than the most-attended word's (a word that appears twice can tie with
itself). Report the share of reviews where the most-attended word has rank 1, the share with rank 3 or
better, and the median rank.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
most-attended word is the most important: 9%
most-attended word in the top 3:          31%
median rank of the most-attended word:    6 (review length: median 193 words)
~~~

The most-attended word is the single most important word in only 9% of the reviews, and among the top 3
in about 30%; its median rank is 6. The weights do carry information: a word chosen at random would rank
around the middle, at about 95, on average. But they are not a faithful account of what the prediction
depends on. Read an attention map as a hint about where a layer looked, not as an explanation of the
decision.
</details>

## Summary

- **Scaled dot-product attention** is a soft lookup: `softmax(Q K^T / sqrt(d_k)) V`. Each query is
  scored against every key, the softmax turns each row of scores into weights that sum to 1, and the
  output blends the values. Step 1 reproduced the concept page's worked example and matched PyTorch's
  built-in to within 1e-6.
- In **self-attention** Q, K and V come from the same sequence. With `Q = K = X` the scores are
  symmetric; **learned projections** `Wq`, `Wk`, `Wv` let a token look for something it is not.
- The **sqrt(d_k)** scale keeps the softmax from saturating: unscaled, the gradient size fell from 0.340
  to 0.082 as `d_k` grew from 4 to 256; scaled, it stayed near 0.36.
- On a **content lookup**, attention scored 1.0000 against 0.3730 for mean pooling. On the **reach**
  task, one attention layer scored 1.00 at every distance; a CNN scored 1.00 only within its receptive
  field (3 or 6 positions) and chance beyond.
- On **IMDB**, the cheap bag of embeddings (0.866) beat attention (0.851) and the CNN (0.840), at a
  fraction of attention's training time. Attention needed a **padding mask** (Ex2), gave identical scores
  to same-word sentences with opposite meanings, costs `L^2` memory, and its weights are a hint, not an
  explanation (Ex3).
- Self-attention is **equivariant** to reordering and an order-free readout is **invariant** (Ex1).
  Next (L22): **positional encoding**, **multi-head** attention, and the residual + layer-norm blocks
  that stack attention into the **Transformer**.
