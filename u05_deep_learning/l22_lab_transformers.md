---
title: "Lab: Build a Transformer, from Heads to a Tiny GPT"
unit: "V"
lesson: "22"
type: lab
tags: [deep-learning, pytorch, transformer, multi-head-attention, positional-encoding, layer-norm, causal-masking, language-model, imdb]
difficulty: intermediate
duration: "90 mins"
---

**Goal:** turn L21's single attention layer into a Transformer, one part at a time, and measure what
each part buys. Part A works with small tensors: **multi-head attention** and the **encoder block**
built from scratch and checked against PyTorch's own layers, **positional encoding**, **layer
normalization** by hand, a small Transformer that solves an order task only when it is told the
positions (and an **ablation** that tests what its heads really contribute), and an experiment on
**depth** that shows why the residual connection matters and where the normalization should go. Part B
moves to real text: an **IMDB** sentiment classifier with and without positional encoding, and a **tiny
GPT** that learns to write movie-review text one character at a time, uses the order of what it reads,
and, in Exercise 4, learns it even without positions.
Pairs with the concept note [The Transformer Architecture](l22_concept_transformers.qmd).

> **Previously:** L21 -- Attention Mechanisms (one attention layer; blind to order)  |  **Next:** L23 -- K-Means Clustering & Similarity (Unit VI)

> This page is the read-only view. To run the lab, open the notebook (`l22_lab_transformers.ipynb`) -- in Colab via the badge below, or locally. **Use a GPU runtime** (Runtime > Change runtime type > T4 GPU): the lab trains about a dozen models, and on a T4 the whole notebook takes 4 to 5 minutes. A CPU also works but takes much longer (20 to 40 minutes on the machine that produced this page). The outputs shown here came from a CPU, so your numbers on a GPU or another machine may differ slightly.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u05_deep_learning/l22_lab_transformers.ipynb)

## Scenario

L21 ended with two sentences made of the same words: "this movie is not bad, it is good" and "this
movie is not good, it is bad". Its attention classifier gave both exactly the same score, 0.8448,
because attention compares words by content and never by position. The **Transformer** keeps that
attention and adds a few parts around it: several attention heads side by side, a position signal, a
residual connection and a normalization around every sublayer, and a stack of blocks of the same design. We
build each part, check it against PyTorch, and test what it changes: first on tensors small enough to
read, then on real movie reviews.

## Setup

Two cells: the first installs, the second imports, seeds and picks the device. The IMDB data is
downloaded in Step 7.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships these, so this is effectively a no-op there.
%pip install -q torch matplotlib pandas pyarrow
# local, in a terminal (not in the notebook):
#   uv add torch matplotlib pandas pyarrow
```

```python
# Setup, cell 2 of 2 -- IMPORTS + SEEDS + DEVICE (safe to re-run)
import math
import re
import copy
import collections
import textwrap
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

## Part A -- The Transformer on Small Tensors

### Step 1: Multi-Head Attention

We start from L21's `attention` function, with one change to the mask. In L21 the mask was a padding
mask of shape `(batch, length)`, and the function reshaped it. Here the caller passes a mask already
shaped to line up with the scores: `True` where a query **may** attend, `False` where it may not. That
one convention covers both masks of this lesson: the padding mask (shape `(B, 1, 1, L)`, the same for
every head and every query) and, in Step 8, the causal mask (shape `(L, L)`). Without a mask the
function is exactly L21's.

**Multi-head attention** runs `h` attentions side by side. Each one (a **head**) projects the
**whole** token, all `d` numbers, down to its own `d/h` queries, keys and values. In code that is one
`d x d` projection per role, cut into `h` slices of `d/h` columns: the same thing, done in one matrix
product. Each head attends on its slice, the `h` results are put side by side again, and a final
linear layer `Wo` mixes them. Inside a head the scores are divided by the square root of `d/h`, not
of `d`; `attention` reads `d_k` from the shape of `Q`, so this happens by itself:

```python
def attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V -> (output, weights).
    mask: True where a query may attend; any shape that
    broadcasts to the scores, e.g. (L, L) or (B, 1, 1, L)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ V, weights

class MultiHeadSelfAttention(nn.Module):
    """h heads in parallel; each projects the whole token
    to d/h numbers (one d x d projection, cut in h)."""
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0, "d must split evenly"
        self.h, self.d_head = n_heads, d // n_heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)

    def split(self, t):
        # (B, L, d) -> (B, h, L, d_head)
        B, L, _ = t.shape
        return t.view(B, L, self.h, self.d_head).transpose(1, 2)

    def forward(self, x, mask=None, return_attn=False):
        B, L, d = x.shape
        q = self.split(self.Wq(x))
        k = self.split(self.Wk(x))
        v = self.split(self.Wv(x))
        ctx, w = attention(q, k, v, mask)   # every head at once
        # concatenate the heads back to width d, then mix
        ctx = ctx.transpose(1, 2).reshape(B, L, d)
        out = self.Wo(ctx)
        return (out, w) if return_attn else out

torch.manual_seed(0)
mha = MultiHeadSelfAttention(d=64, n_heads=4)
x = torch.randn(2, 7, 64)
out, w = mha(x, return_attn=True)
print("input", tuple(x.shape), "-> output", tuple(out.shape))
print("weights", tuple(w.shape), "= (batch, head, query, key)")
print("each head projects all 64 numbers down to", mha.d_head)
```

~~~text
input (2, 7, 64) -> output (2, 7, 64)
weights (2, 4, 7, 7) = (batch, head, query, key)
each head projects all 64 numbers down to 16
~~~

Is this the same computation as PyTorch's `nn.MultiheadAttention`? Copy our weights into it and
compare the outputs. PyTorch stores the three input projections as one stacked matrix, `in_proj_weight`,
with `Wq` on top, then `Wk`, then `Wv`:

```python
ref = nn.MultiheadAttention(64, 4, batch_first=True)
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat(
        [mha.Wq.weight, mha.Wk.weight, mha.Wv.weight]))
    ref.in_proj_bias.copy_(torch.cat(
        [mha.Wq.bias, mha.Wk.bias, mha.Wv.bias]))
    ref.out_proj.weight.copy_(mha.Wo.weight)
    ref.out_proj.bias.copy_(mha.Wo.bias)
    ref_out, ref_w = ref(x, x, x, average_attn_weights=False)
print("outputs match nn.MultiheadAttention:",
      torch.allclose(out, ref_out, atol=1e-6))
print("weights match:", torch.allclose(w, ref_w, atol=1e-6))
```

~~~text
outputs match nn.MultiheadAttention: True
weights match: True
~~~

Both match to within `1e-6`, so our class computes what PyTorch's built-in computes; from here on you
can read `nn.MultiheadAttention` as "the class above, written for speed".

**What do the heads cost?** Each head's queries, keys and values have `d/h` numbers, so the matrix
products of four heads over 16 numbers add up to the same arithmetic as one head over 64. The softmax is different: it now
runs over `h` matrices of `L x L` scores instead of one. So is the memory, in an implementation like
ours that stores the weights: every head keeps its own `L x L` weight matrix. Measure it for one
sequence of 1,024 tokens:

```python
xl = torch.randn(1, 1024, 64)
for n_heads in [1, 4, 16]:
    torch.manual_seed(0)
    m = MultiHeadSelfAttention(64, n_heads)
    with torch.no_grad():
        _, wl = m(xl, return_attn=True)
    mb = wl.nelement() * wl.element_size() / 2**20
    print(f"{n_heads:2d} heads: weights {tuple(wl.shape)},"
          f" {mb:5.1f} MB")
```

~~~text
 1 heads: weights (1, 1, 1024, 1024),   4.0 MB
 4 heads: weights (1, 4, 1024, 1024),  16.0 MB
16 heads: weights (1, 16, 1024, 1024),  64.0 MB
~~~

The weights grow with the number of heads: `h` heads hold `h` matrices of `L x L` scores. That is L21's
quadratic memory, multiplied by `h`. Fused kernels such as FlashAttention (L21), which PyTorch's
`F.scaled_dot_product_attention` uses on a GPU when it can, compute the same result in tiles and never
store the full matrices; the arithmetic still grows with `L x L`.

<details><summary>(Optional) Reading the code: <code>view</code> + <code>transpose</code>, <code>reshape</code>, <code>copy_</code></summary>

**`t.view(B, L, self.h, self.d_head).transpose(1, 2)`.** `view` cuts each token's 64 numbers into 4
groups of 16 without copying anything: `(B, L, 64)` becomes `(B, L, 4, 16)`. `transpose(1, 2)` swaps the
length and head axes to give `(B, 4, L, 16)`, so the head axis sits next to the batch axis and `attention`
runs every head of every sequence in one batched matrix product (L21's `@` with extra leading axes).

**`ctx.transpose(1, 2).reshape(B, L, d)`** undoes the split: back to `(B, L, 4, 16)`, then the last two
axes are glued into 64. After a `transpose` the numbers are no longer laid out in order in memory, so
`view` would fail here; `reshape` copies when it has to.

**`ref.in_proj_weight.copy_(...)`** writes our numbers into PyTorch's tensor in place. The trailing
underscore marks an in-place operation in PyTorch; it runs inside `torch.no_grad()` because writing into a
parameter must not be recorded for backpropagation.
</details>

### Step 2: Positional Encoding

Self-attention without a causal mask treats its input as a set. To tell it where each token sits, add a **position signal** to
each token's vector before the first block. The original Transformer uses fixed **sinusoidal**
encodings: dimension pair `(2i, 2i+1)` holds the sine and cosine of the position at a frequency that
falls geometrically with `i`. The concept page works out the first three positions with `d = 4` by
hand; here they are, followed by the real size:

```python
def sinusoidal_pos_encoding(max_len, d):
    """Vaswani et al. (2017): sines and cosines of the
    position at geometrically spaced frequencies."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    i = torch.arange(0, d, 2).float()
    freq = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe

print("d = 4, positions 0, 1, 2 (concept page):")
print(sinusoidal_pos_encoding(3, 4).numpy())
pe = sinusoidal_pos_encoding(max_len=16, d=64)
print("PE for 16 positions, width 64:", tuple(pe.shape))

same = torch.ones(3, 64)      # three IDENTICAL token vectors
print("identical without PE:", torch.equal(same[0], same[1]))
print("identical with PE:   ",
      torch.equal((same + pe[:3])[0], (same + pe[:3])[1]))
```

~~~text
d = 4, positions 0, 1, 2 (concept page):
[[ 0.     1.     0.     1.   ]
 [ 0.841  0.54   0.01   1.   ]
 [ 0.909 -0.416  0.02   1.   ]]
PE for 16 positions, width 64: (16, 64)
identical without PE: True
identical with PE:    False
~~~

Without the encoding, three copies of the same token are indistinguishable, so attention would treat
them identically wherever they sit; after adding it, each copy carries its position. The whole encoding
matrix is a set of stacked waves:

```python
plt.figure(figsize=(6, 3))
plt.imshow(pe.numpy().T, aspect="auto", cmap="RdBu_r",
           vmin=-1, vmax=1, origin="lower")
plt.xlabel("position"); plt.ylabel("encoding dimension")
plt.title("sinusoidal positional encoding")
plt.colorbar(); plt.show()
```

The first dimensions oscillate fast and tell neighbors apart; the last ones change slowly and tell
distant positions apart. Together they give every position a distinct code.

### Step 3: Layer Normalization

**Layer normalization** standardizes each token **across its own features**: subtract the token's mean,
divide by its standard deviation. The concept page does it by hand for the token `[2, 4, 6, 8]`; compare
with `nn.LayerNorm`:

```python
tok = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
mean = tok.mean()
var = tok.var(unbiased=False)
by_hand = (tok - mean) / torch.sqrt(var + 1e-5)
ln = nn.LayerNorm(4)
print(f"mean {mean.item():.1f}, variance {var.item():.1f}")
print("by hand:     ", by_hand.numpy())
print("nn.LayerNorm:", ln(tok).detach().numpy())
```

~~~text
mean 5.0, variance 5.0
by hand:      [[-1.342 -0.447  0.447  1.342]]
nn.LayerNorm: [[-1.342 -0.447  0.447  1.342]]
~~~

`nn.LayerNorm` also has a learned scale and shift per feature (`ln.weight`, `ln.bias`), which start at
1 and 0, so a fresh layer gives exactly the hand result. The small `1e-5` (PyTorch's `eps`) guards
against dividing by zero.

Now the difference from L17's **batch normalization**, which standardizes each feature **across the
batch**. Normalize one sequence on its own, then the same sequence inside a batch of three:

```python
torch.manual_seed(0)
X = torch.randn(3, 5, 4) * 3 + 7   # (batch, tokens, features)
Y = ln(X).detach()
print("per-token mean: largest |mean|",
      f"{Y.mean(-1).abs().max().item():.3f}")
print("per-token std:  from",
      f"{Y.std(-1, unbiased=False).min().item():.3f} to",
      f"{Y.std(-1, unbiased=False).max().item():.3f}")

bn = nn.BatchNorm1d(4)   # normalizes each feature over the batch
def bn_tokens(t):
    return bn(t.transpose(1, 2)).transpose(1, 2)
with torch.no_grad():
    print("layer norm: sequence 0 alone == in the batch:",
          torch.allclose(ln(X[:1]), ln(X)[:1]))
    print("batch norm: sequence 0 alone == in the batch:",
          torch.allclose(bn_tokens(X[:1]), bn_tokens(X)[:1]))
```

~~~text
per-token mean: largest |mean| 0.000
per-token std:  from 1.000 to 1.000
layer norm: sequence 0 alone == in the batch: True
batch norm: sequence 0 alone == in the batch: False
~~~

Every token comes out with mean 0 and standard deviation 1. Layer norm gives sequence 0 the same result
alone or in a batch, because it never looks at the other sequences; batch norm (in training mode)
does not, because its statistics come from whatever else is in the batch. That independence is why
Transformers, which see sequences of different lengths in batches of any size, use layer norm.

### Step 4: The Encoder Block

The block wraps two sublayers the same way. **Multi-head self-attention** mixes information between
positions; a **position-wise feed-forward network** (a two-layer MLP applied to each position on its
own) transforms each position. Around each sublayer sit a **residual connection** (L19: add the input
back) and a **layer norm**. The original Transformer normalizes *after* adding the residual, called
**Post-LN**; GPT-2 and most later models normalize the sublayer's *input*, called **Pre-LN**. The class
below does both, plus a switch that removes the residual, for the experiment in Step 6:

```python
class EncoderBlock(nn.Module):
    """self-attention -> Add & Norm -> FFN -> Add & Norm."""
    def __init__(self, d, n_heads, d_ff, norm_first=False,
                 residual=True):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d, n_heads)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(),
                                nn.Linear(d_ff, d))
        self.norm2 = nn.LayerNorm(d)
        self.norm_first = norm_first
        # Step 6 switches the residual off (keep = 0)
        self.keep = 1.0 if residual else 0.0

    def forward(self, x, mask=None):
        if self.norm_first:     # Pre-LN (GPT-2 and later)
            x = self.keep * x + self.attn(self.norm1(x), mask)
            x = self.keep * x + self.ff(self.norm2(x))
        else:                   # Post-LN (the 2017 paper)
            x = self.norm1(self.keep * x + self.attn(x, mask))
            x = self.norm2(self.keep * x + self.ff(x))
        return x

torch.manual_seed(0)
block = EncoderBlock(d=64, n_heads=4, d_ff=128)
h = block(torch.randn(2, 12, 64))
print("block: (2, 12, 64) ->", tuple(h.shape))
```

~~~text
block: (2, 12, 64) -> (2, 12, 64)
~~~

A block maps a sequence to a sequence of the same shape, so blocks stack.

```python
# Your turn (warm-up): the block has 4 projections Linear(64, 64),
# two LayerNorm(64), and a feed-forward Linear(64, 128) +
# Linear(128, 64). Predict its parameter count, then check.
# Uncomment and fill the blank:
# n = sum(p.numel() for p in block.parameters())
# print("block parameters:", n)   # expect ____
```

The block has **33,472** parameters: `4 * (64*64 + 64) = 16,640` in the attention projections,
`(64*128 + 128) + (128*64 + 64) = 16,576` in the feed-forward network, and `2 * (2*64) = 256` in the two
layer norms (a scale and a shift per feature). The two sublayers hold almost all of it, in nearly equal
parts.

PyTorch's `nn.TransformerEncoderLayer` is the same block. With dropout off and our weights copied in, it
gives the same output:

```python
tel = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0,
                                 batch_first=True)
a = block.attn
with torch.no_grad():
    sa = tel.self_attn
    sa.in_proj_weight.copy_(torch.cat(
        [a.Wq.weight, a.Wk.weight, a.Wv.weight]))
    sa.in_proj_bias.copy_(torch.cat(
        [a.Wq.bias, a.Wk.bias, a.Wv.bias]))
    sa.out_proj.weight.copy_(a.Wo.weight)
    sa.out_proj.bias.copy_(a.Wo.bias)
    for ours, theirs in [(block.ff[0], tel.linear1),
                         (block.ff[2], tel.linear2),
                         (block.norm1, tel.norm1),
                         (block.norm2, tel.norm2)]:
        theirs.weight.copy_(ours.weight)
        theirs.bias.copy_(ours.bias)
    x2 = torch.randn(2, 12, 64)
    diff = (block(x2) - tel.eval()(x2)).abs().max().item()
print("largest difference from nn.TransformerEncoderLayer:",
      f"{diff:.1e}")
print("same parameter count:",
      sum(p.numel() for p in tel.parameters()))
```

~~~text
largest difference from nn.TransformerEncoderLayer: 0.0e+00
same parameter count: 33472
~~~

Its `norm_first=True` option switches it to Pre-LN, like ours.

### Step 5: Train a Transformer on Order

Now a task where order is the **only** signal. Each sequence is a `CLS` token at position 0 followed
by filler tokens, with one marker **A** and one marker **B** at two random positions. The label is **1 if
A comes before B**. Both classes contain exactly the same tokens; only their order differs. The
classifier reads its answer from the `CLS` position, like BERT.

```python
def make_order_dataset(n, seq_len=12, vocab=10, seed=0):
    """'Does A come before B?' CLS at position 0, one A
    (token 1) and one B (token 2) at two random positions,
    fillers (3..9) elsewhere. The token MULTISET is the same
    either way, so order is the only signal."""
    g = torch.Generator().manual_seed(seed)
    CLS, A, B = 0, 1, 2
    x = torch.zeros(n, seq_len, dtype=torch.long)
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        seq = torch.randint(3, vocab, (seq_len,), generator=g)
        seq[0] = CLS
        p = torch.randperm(seq_len - 1, generator=g)[:2] + 1
        pa, pb = p[0].item(), p[1].item()
        seq[pa], seq[pb] = A, B
        x[i] = seq
        y[i] = 1 if pa < pb else 0
    return x, y

xtr, ytr = make_order_dataset(4000, seed=0)
xte, yte = make_order_dataset(1000, seed=1)
print("train", tuple(xtr.shape), "| test", tuple(xte.shape))
print("example 0:", xtr[0].tolist(), "-> y =", ytr[0].item())
print("test share of label 1:",
      f"{yte.float().mean().item():.3f}")
xtr, ytr = xtr.to(device), ytr.to(device)
xte, yte = xte.to(device), yte.to(device)
```

~~~text
train (4000, 12) | test (1000, 12)
example 0: [0, 6, 3, 1, 4, 5, 9, 6, 7, 8, 5, 2] -> y = 1
test share of label 1: 0.517
~~~

In example 0, A (token `1`) sits at position 3 and B (token `2`) at position 11, so the label is 1. In the
test set 51.7% of the labels are 1, so a model that always answers 1 scores 0.517: that is the bar for
"learned nothing". The model is a token embedding, an optional position signal, a stack of encoder
blocks, and a linear classifier on the `CLS` position:

```python
class TinyTransformer(nn.Module):
    def __init__(self, vocab=10, d=64, n_heads=4, n_blocks=2,
                 d_ff=128, max_len=16, pos_mode="sinusoidal",
                 norm_first=False, residual=True):
        super().__init__()
        self.pos_mode = pos_mode
        self.tok_emb = nn.Embedding(vocab, d)
        if pos_mode == "sinusoidal":
            self.register_buffer(
                "pe", sinusoidal_pos_encoding(max_len, d))
        elif pos_mode == "learned":
            self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d, n_heads, d_ff, norm_first, residual)
             for _ in range(n_blocks)])
        self.head = nn.Linear(d, 2)

    def embed(self, x):
        """token vectors plus the position signal"""
        L = x.shape[1]
        h = self.tok_emb(x)
        if self.pos_mode == "sinusoidal":
            h = h + self.pe[:L]
        elif self.pos_mode == "learned":
            h = h + self.pos_emb(torch.arange(L, device=x.device))
        return h

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, 0])        # read out at CLS

def train(model, epochs=15, bs=256, lr=1e-3):
    """Adam + cross-entropy on the order task -> test acc."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(epoch)
        perm = torch.randperm(len(ytr), generator=g)
        model.train()
        for s in range(0, len(ytr), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss_fn(model(xtr[b]), ytr[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xte).argmax(1)
    return (pred == yte).float().mean().item()

torch.manual_seed(0)
no_pos = TinyTransformer(pos_mode="none")
acc_no = train(no_pos)
torch.manual_seed(0)
with_pos = TinyTransformer(pos_mode="sinusoidal")
acc_pe = train(with_pos)
print(f"always answer 1:          {yte.float().mean().item():.4f}")
print(f"NO positional encoding:   {acc_no:.4f}")
print(f"WITH positional encoding: {acc_pe:.4f}")
```

~~~text
always answer 1:          0.5170
NO positional encoding:   0.4810
WITH positional encoding: 1.0000
~~~

<details><summary>(Optional) Reading the code: <code>register_buffer</code>, <code>nn.ModuleList</code></summary>

**`self.register_buffer("pe", ...)`** stores a tensor in the module without making it a parameter:
it moves with `model.to(device)` and is saved with the model, but Adam never updates it. The sinusoidal
encoding is fixed, so it is a buffer; the learned encoding (`nn.Embedding`) is a parameter.

**`nn.ModuleList([...])`** holds a list of layers so that PyTorch sees their parameters. A plain Python
list of blocks would train nothing: `model.parameters()` would not find them.
</details>

Same architecture, same data, same training. Without positional encoding the model scores 0.4810, no
better than always answering 1 (the exact figure varies a little between machines: 0.478 to 0.481 in our
tests); with it, the model is perfect. Shuffle the positions of one test sequence (keeping `CLS` first) and look at the logits:

```python
torch.manual_seed(1)
perm = torch.cat([torch.tensor([0]), torch.randperm(11) + 1])
with torch.no_grad():
    a, a_shuf = no_pos(xte[:1]), no_pos(xte[:1, perm])
    c, c_shuf = with_pos(xte[:1]), with_pos(xte[:1, perm])
print("no-PE logits unchanged by the shuffle:",
      torch.allclose(a, a_shuf, atol=1e-5))
print("PE logits unchanged by the shuffle:   ",
      torch.allclose(c, c_shuf, atol=1e-5))
```

~~~text
no-PE logits unchanged by the shuffle: True
PE logits unchanged by the shuffle:    False
~~~

The model without positions gives the same logits however the other tokens are arranged (up to
rounding): self-attention is permutation-equivariant (L21), and the `CLS` readout ignores where the
others sit, so "A before B" and "B before A" look identical to it. Adding the position signal breaks
that symmetry. (The shuffle keeps `CLS` at position 0. Moving it would change which token the readout
reads, and so the logits, even without positions. And all of this holds for self-attention without a
causal mask; Exercise 4 shows that a causal decoder is a different story.)

**What do the trained heads do?** For the `CLS` query of the last block, average each head's attention
weight on the A position and on the B position over the 1,000 test sequences:

```python
@torch.no_grad()
def attention_maps(model, x):
    """each block's attention weights, (n, head, L, L)"""
    model.eval()
    h, maps = model.embed(x), []
    for blk in model.blocks:
        _, w = blk.attn(h, return_attn=True)
        maps.append(w.cpu())
        h = blk(h)
    return maps

w_last = attention_maps(with_pos, xte)[-1]
cls_row = w_last[:, :, 0, :]              # (n, head, key)
xc = xte.cpu()
on_a = (cls_row * (xc == 1)[:, None, :]).sum(-1).mean(0)
on_b = (cls_row * (xc == 2)[:, None, :]).sum(-1).mean(0)
print("CLS query, last block -- average weight")
print("head   on A    on B")
for k in range(4):
    print(f"{k:4d}  {on_a[k].item():5.3f}  {on_b[k].item():5.3f}")

fig, axes = plt.subplots(4, 1, figsize=(7, 4.2), sharex=True)
toks = ["CLS" if t == 0 else "A" if t == 1 else "B" if t == 2
        else str(t) for t in xc[0].tolist()]
for k, ax in enumerate(axes):
    ax.imshow(cls_row[0, k][None].numpy(), cmap="Blues",
              vmin=0, vmax=1, aspect="auto")
    ax.set_yticks([0]); ax.set_yticklabels([f"head {k}"])
axes[-1].set_xticks(range(12)); axes[-1].set_xticklabels(toks)
fig.suptitle("CLS attention in the last block, test sequence 0")
plt.tight_layout(); plt.show()
```

~~~text
CLS query, last block -- average weight
head   on A    on B
   0  0.998  0.000
   1  0.002  0.972
   2  0.055  0.364
   3  0.338  0.558
~~~

The heads are not copies of each other. Averaged over the test set, head 0 puts 0.998 of the `CLS`
token's attention on A and head 1 puts 0.972 on B, the two facts the answer depends on. On these
averages, heads 2 and 3 look less decisive (head 3 leans toward B). The plot shows one test sequence,
where B comes before A (label 0): head 0 sits on A, heads 1 and 3 on B.

It is tempting to conclude that head 0 finds A and head 1 finds B. The weights cannot tell us that:
they show where a head looks, not whether the answer needs what it finds. Test it directly. Switch
off one head of the last block, by zeroing the 16 columns of `Wo` through which its output enters the
mix, and measure the test accuracy; then switch off the other three and keep only that one. The table
also shows each head's average weight on whichever marker comes **first** in the sequence:

```python
def drop_heads(model, heads):
    """a copy of model with these last-block heads off"""
    # deepcopy: an independent copy, so with_pos is
    # left untouched
    m = copy.deepcopy(model)
    Wo = m.blocks[-1].attn.Wo.weight
    with torch.no_grad():
        for k in heads:
            # head k's 16 numbers enter Wo through
            # columns 16k to 16k + 15
            Wo[:, k * 16:(k + 1) * 16] = 0
    return m

@torch.no_grad()
def accuracy(model):
    pred = model(xte).argmax(1)
    return (pred == yte).float().mean().item()

# weight on whichever marker comes first
pos_a = (xc == 1).float().argmax(1)
pos_b = (xc == 2).float().argmax(1)
first = torch.minimum(pos_a, pos_b)
on_1st = cls_row[torch.arange(len(xc)), :, first].mean(0)

print("head  on A   on B   on 1st  without it  only it")
for k in range(4):
    others = [j for j in range(4) if j != k]
    a_wo = accuracy(drop_heads(with_pos, [k]))
    a_only = accuracy(drop_heads(with_pos, others))
    print(f"{k:4d}  {on_a[k].item():5.3f}  "
          f"{on_b[k].item():5.3f}  {on_1st[k].item():6.3f}"
          f"  {a_wo:10.4f}  {a_only:7.4f}")
print("all four heads of the last block off:",
      f"{accuracy(drop_heads(with_pos, range(4))):.4f}")
```

~~~text
head  on A   on B   on 1st  without it  only it
   0  0.998  0.000   0.516      0.9960   0.8750
   1  0.002  0.972   0.485      1.0000   0.9760
   2  0.055  0.364   0.105      1.0000   0.9210
   3  0.338  0.558   0.791      0.9420   1.0000
all four heads of the last block off: 0.9110
~~~

The B head can go at no cost at all. The head the model misses most is head 3, whose A and B averages
looked mixed, and head 3 on its own is enough. The third column shows why: sequence by sequence, head 3
looks at **whichever marker comes first**, exactly the relation the label asks about, and averaging
over A and B hid it. The heads overlap, and a large attention weight does not say which head the answer
depends on. That is L21's warning, now measured: **attention weights are a hint, not an explanation**.
Switching parts off and measuring (an **ablation**) is how you test what a part contributes. The roles
also depend on the random start. In our runs with seeds 1 and 2 the last block still had sharp heads
(seed 2: 0.965 on B), yet switching off all four of them cost almost nothing (0.996 and 0.984), while
switching off the first block's heads did hurt (0.483 and 0.711).

### Step 6: Depth -- the Residual and Where the Norm Goes

Real Transformers stack dozens of blocks. Train the same order task at two depths: 6 blocks with and
without the residual connection, and 12 blocks with Post-LN and with Pre-LN. Nothing else changes (Adam,
learning rate 0.001, 15 epochs):

```python
runs = [(6, "Post-LN + residual", False, True),
        (6, "Post-LN, NO residual", False, False),
        (12, "Post-LN + residual", False, True),
        (12, "Pre-LN + residual", True, True)]
print("blocks  variant                test acc")
for n_blocks, name, norm_first, residual in runs:
    torch.manual_seed(0)
    m = TinyTransformer(n_blocks=n_blocks, norm_first=norm_first,
                        residual=residual)
    print(f"{n_blocks:6d}  {name:21s}  {train(m):.4f}")
```

~~~text
blocks  variant                test acc
     6  Post-LN + residual     1.0000
     6  Post-LN, NO residual   0.4830
    12  Post-LN + residual     0.4830
    12  Pre-LN + residual      1.0000
~~~

**0.4830 is the share of label 0 in the test set**: those models answer 0 for every one of the 1,000
sequences. They never got started. With 6 blocks, the residual connection is the whole difference
between a perfect model and a stuck one: it gives every block a direct path from input to output, so
the signal and the gradient do not have to pass through six freshly initialized sublayers in a row.
At 12 blocks the residual is not enough for Post-LN, while Pre-LN learns the task perfectly. In our
tests with seeds 0, 1 and 2, the 12-block Post-LN model was stuck in two runs of three (the third
reached 0.968) and Pre-LN reached 1.0000 in all three; the 6-block model without a residual was stuck
in all three. Post-LN is not broken: the 2017 paper trained it with a learning-rate **warm-up**, and
Exercise 2 shows that this rescues it. Here Pre-LN needed none; most models since GPT-2 put the norm
first.

## Part B -- Real Text: IMDB Reviews

### Step 7: A Transformer Encoder for Sentiment, With and Without Position

L21 trained three sentiment classifiers on 25,000 IMDB reviews. The best was the cheapest, a bag of
embeddings (0.8662), ahead of self-attention (0.8514), and both order-blind models (the bag and
self-attention) gave the two same-word sentences identical scores; only the CNN, which reads 5-word
windows, told them apart. Does a Transformer with positional encoding do better? The data pipeline is L21's,
unchanged: the same pinned download (two Parquet files of about 21 MB from the Hugging Face Hub; the
dataset card lists the licence as "other" without further terms, so cite Maas et al., 2011, and treat
the reviews, written by IMDb users, as material for study, not for redistribution), the same word
tokenizer and 20,000-word vocabulary, reviews cut at 256 words, and the same 10,000-review test sample.

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

test_sub = test_df.sample(frac=1, random_state=1).iloc[:10000]
train_sh = train_df.sample(frac=1, random_state=0)
Xtr_txt = encode(train_sh["text"])
ytr_txt = torch.tensor(train_sh["label"].values)
Xte_txt = encode(test_sub["text"])
yte_txt = torch.tensor(test_sub["label"].values)
print("Xtr_txt", tuple(Xtr_txt.shape), "| Xte_txt",
      tuple(Xte_txt.shape))
```

~~~text
Xtr_txt (25000, 256) | Xte_txt (10000, 256)
~~~

The classifier puts a `CLS` token in front of every review (a new id, 20,000, after the vocabulary),
adds the position signal, runs two encoder blocks with a **padding mask** (shape `(B, 1, 1, L)`: no query
may attend to a padded key), and reads the answer from `CLS`. We train it twice, without and with
sinusoidal positional encoding:

```python
CLS_ID = VOCAB

class TextTransformer(nn.Module):
    def __init__(self, pos_mode, d=64, n_heads=4, n_blocks=2,
                 d_ff=128, max_len=MAX_LEN + 1):
        super().__init__()
        self.pos_mode = pos_mode
        self.emb = nn.Embedding(VOCAB + 1, d, padding_idx=0)
        if pos_mode == "sinusoidal":
            self.register_buffer(
                "pe", sinusoidal_pos_encoding(max_len, d))
        self.blocks = nn.ModuleList(
            [EncoderBlock(d, n_heads, d_ff)
             for _ in range(n_blocks)])
        self.out = nn.Linear(d, 2)

    def forward(self, x):
        cls = torch.full((len(x), 1), CLS_ID, device=x.device)
        x = torch.cat([cls, x], 1)
        mask = (x != 0)[:, None, None, :]   # (B, 1, 1, L)
        h = self.emb(x)
        if self.pos_mode == "sinusoidal":
            h = h + self.pe[:x.shape[1]]
        for blk in self.blocks:
            h = blk(h, mask)
        return self.out(h[:, 0])

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
def predict_proba(model, X, bs=500):
    """P(positive) for every row of X."""
    model.eval()
    out = [model(X[s:s + bs].to(device)).softmax(-1)[:, 1]
           for s in range(0, len(X), bs)]
    return torch.cat(out).cpu()

text_models = {}
for name in ["none", "sinusoidal"]:
    torch.manual_seed(0)
    text_models[name] = fit_text(TextTransformer(name))

print(f"{'positional encoding':20s} {'params':>9} {'test acc':>9}")
for name, model in text_models.items():
    n_par = sum(p.numel() for p in model.parameters())
    pred = (predict_proba(model, Xte_txt) > 0.5).long()
    acc = (pred == yte_txt).float().mean().item()
    print(f"{name:20s} {n_par:>9,} {acc:>9.4f}")
```

~~~text
positional encoding     params  test acc
none                 1,347,138    0.8483
sinusoidal           1,347,138    0.8483
~~~

Positional encoding changes nothing measurable here: both models score 0.8483 on the 10,000 test
reviews (with other seeds and on other machines, the version with positions scored the same or lower,
by up to about 0.016, never higher). Both sit a little below L21's bag of embeddings
(0.8662), which trained in seconds; each Transformer takes about a minute on a T4.

Now the question L21 left open. Eight pairs of sentences, each pair made of exactly the same words;
`sign` says which sentence of the pair is the more positive one (+1: the first):

```python
pairs = [
    ("this movie is not bad , it is good",
     "this movie is not good , it is bad", +1),
    ("the acting was not great , it was terrible",
     "the acting was not terrible , it was great", -1),
    ("i expected a bad film but it was a great film",
     "i expected a great film but it was a bad film", +1),
    ("the story is boring but the ending is wonderful",
     "the story is wonderful but the ending is boring", +1),
    ("not a good movie , a bad one",
     "not a bad movie , a good one", -1),
    ("i thought it would be awful , it was excellent",
     "i thought it would be excellent , it was awful", +1),
    ("the first half is bad , the rest is brilliant",
     "the first half is brilliant , the rest is bad", +1),
    ("never boring , always fun",
     "never fun , always boring", +1)]
first = [p[0] for p in pairs]
second = [p[1] for p in pairs]
print("P(positive) for the 1st and 2nd sentence of each pair")
right = 0
for i, (s1, s2, sign) in enumerate(pairs):
    p = {}
    for name, model in text_models.items():
        pr = predict_proba(model, encode([first[i], second[i]]))
        p[name] = pr.tolist()
    gap = p["sinusoidal"][0] - p["sinusoidal"][1]
    right += int(gap * sign > 0)
    print(f"{i + 1}. {s1}\n   {s2}")
    print(f"   no PE: {p['none'][0]:.3f} {p['none'][1]:.3f}"
          f"   PE: {p['sinusoidal'][0]:.3f}"
          f" {p['sinusoidal'][1]:.3f}")
print(f"PE: the more positive sentence scored higher in {right}/8")
```

~~~text
P(positive) for the 1st and 2nd sentence of each pair
1. this movie is not bad , it is good
   this movie is not good , it is bad
   no PE: 0.008 0.008   PE: 0.023 0.011
2. the acting was not great , it was terrible
   the acting was not terrible , it was great
   no PE: 0.047 0.047   PE: 0.024 0.025
3. i expected a bad film but it was a great film
   i expected a great film but it was a bad film
   no PE: 0.906 0.906   PE: 0.601 0.472
4. the story is boring but the ending is wonderful
   the story is wonderful but the ending is boring
   no PE: 0.014 0.014   PE: 0.072 0.023
5. not a good movie , a bad one
   not a bad movie , a good one
   no PE: 0.005 0.005   PE: 0.004 0.005
6. i thought it would be awful , it was excellent
   i thought it would be excellent , it was awful
   no PE: 0.249 0.249   PE: 0.957 0.976
7. the first half is bad , the rest is brilliant
   the first half is brilliant , the rest is bad
   no PE: 0.015 0.015   PE: 0.015 0.013
8. never boring , always fun
   never fun , always boring
   no PE: 0.983 0.983   PE: 0.997 0.997
PE: the more positive sentence scored higher in 6/8
~~~

Without positional encoding, the two sentences of each pair get **identical** scores, as in L21: same
words, same answer. With it, every pair gets two different scores, so the model can now tell the two
orders apart. But it has not learned what the order *means*. The more positive sentence scored higher in
6 of the 8 pairs (in our runs with other seeds and on a GPU, anywhere from 2 to 7 of 8), most
differences are tiny (0.024 against 0.025), and in only one pair do the two sentences land on opposite sides of 0.5 (0.601 / 0.472). Both
"this movie is not bad, it is good" and its twin are still called negative.

Treat these eight pairs with care: they are short, made-up sentences, unlike the reviews the models
trained on, and their scores sit near 0 or 1, where small changes mean little. Two order-blind models
can disagree completely on them: L21's attention model gave "this movie is not bad, it is good"
0.8448, this one 0.008.

So, under these conditions (a small model, trained from scratch for 5 epochs on 25,000 labeled
reviews), position did not help. The likely reason is the training signal. A review's label is decided
mostly by *which* words it contains (that is why L21's bag of embeddings won), and each review of about
230 words (median 173) gives the model a single label to learn from, so little pushes it to use the
positions it is given. **Positional encoding makes order available; the training objective decides
whether the model uses it.** Step 8 uses an objective that gives a target at every position and cannot
be met without order.

### Step 8: A Tiny GPT on Review Text

A classifier trained on 25,000 labels has little reason to learn word order. A **language model** has
every reason: its task is to predict the **next** symbol, at every position, and the next symbol depends
on what came before and in what order. Here is a tiny one, GPT-style: it reads review text one
**character** at a time (no word vocabulary needed), and learns from the first 4,000 training reviews.

```python
raw = " ".join(train_df["text"].iloc[:4000])
text = re.sub(r"[^a-z0-9 .,!?'\-]", "",
              raw.lower().replace("<br />", " "))
text = re.sub(r" +", " ", text)
chars = sorted(set(text))
ch2id = {c: i for i, c in enumerate(chars)}
data = torch.tensor([ch2id[c] for c in text])
n_tr = int(0.9 * len(data))
train_ids, val_ids = data[:n_tr], data[n_tr:]
print(f"{len(text):,} characters, {len(chars)} distinct:",
      "".join(chars))
print("train", len(train_ids), "| validation", len(val_ids))
```

~~~text
5,026,211 characters, 43 distinct:  !',-.0123456789?abcdefghijklmnopqrstuvwxyz
train 4523589 | validation 502622
~~~

The model is a **decoder**: token and learned position embeddings, a stack of Pre-LN blocks whose
self-attention carries a **causal mask** (position `i` may attend to positions `0..i` only), a final
layer norm, and a linear layer that turns every position's vector into scores over the 43 characters.
The blocks are Step 4's `EncoderBlock` class, unchanged: a decoder-only block *is* the encoder block
with a causal mask. The final layer norm is there because in a Pre-LN stack the main residual path is
never normalized, so its scale drifts from block to block; one norm at the top fixes it before the
output layer (GPT-2 does the same). The `pos` switch lets Exercise 4 train the same model without
positions. Each training window of 64 characters gives 64 next-character examples at once: position
`i` must predict character `i + 1`.

```python
CTX = 64

class TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, n_blocks=4,
                 d_ff=512, causal=True, pos=True):
        super().__init__()
        self.causal, self.pos = causal, pos
        self.tok_emb = nn.Embedding(vocab, d)
        # created even when pos=False, so both versions
        # start from the same random weights
        self.pos_emb = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d, n_heads, d_ff, norm_first=True)
             for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def forward(self, x, causal=None):
        causal = self.causal if causal is None else causal
        L = x.shape[1]
        mask = None
        if causal:   # lower triangle: True = may attend
            mask = torch.tril(torch.ones(L, L, dtype=torch.bool,
                                         device=x.device))
        h = self.tok_emb(x)
        if self.pos:
            pos = torch.arange(L, device=x.device)
            h = h + self.pos_emb(pos)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.out(self.norm(h))   # (B, L, vocab)

def get_batch(ids, bs, g):
    """bs random windows; y is x shifted by one character"""
    starts = torch.randint(len(ids) - CTX - 1, (bs,), generator=g)
    x = torch.stack([ids[s:s + CTX] for s in starts])
    y = torch.stack([ids[s + 1:s + CTX + 1] for s in starts])
    return x.to(device), y.to(device)

@torch.no_grad()
def val_loss(model, causal):
    """mean next-character cross-entropy on validation"""
    model.eval()
    g = torch.Generator().manual_seed(123)
    losses = []
    for _ in range(20):
        x, y = get_batch(val_ids, 128, g)
        logits = model(x, causal=causal)
        losses.append(F.cross_entropy(
            logits.flatten(0, 1), y.flatten()).item())
    return sum(losses) / len(losses)

def train_gpt(causal, pos=True, steps=1500, bs=64):
    torch.manual_seed(0)
    model = TinyGPT(len(chars), causal=causal,
                    pos=pos).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    model.train()
    for step in range(steps):
        x, y = get_batch(train_ids, bs, g)
        loss = F.cross_entropy(model(x).flatten(0, 1),
                               y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    return model

gpt = train_gpt(causal=True)
print(f"parameters: {sum(p.numel() for p in gpt.parameters()):,}")
print(f"validation loss: {val_loss(gpt, causal=True):.3f}"
      f"   (uniform guess: ln {len(chars)} ="
      f" {math.log(len(chars)):.3f})")
```

~~~text
parameters: 812,587
validation loss: 1.544   (uniform guess: ln 43 = 3.761)
~~~

A loss of 1.544 is far below the 3.761 of a uniform guess over 43 characters. Now **generate**: start from a
prompt, predict the next character, sample it, append it, and repeat. The model only ever sees the past,
which is exactly what it was trained for:

```python
@torch.no_grad()
def generate(model, prompt, n=200, temperature=0.8, seed=0):
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([[ch2id[c] for c in prompt]],
                       device=device)
    for _ in range(n):
        logits = model(ids[:, -CTX:], causal=True)[:, -1]
        probs = (logits / temperature).softmax(-1)
        nxt = torch.multinomial(probs, 1, generator=g)
        ids = torch.cat([ids, nxt], 1)
    return "".join(chars[i] for i in ids[0].tolist())

print(textwrap.fill(generate(gpt, "this movie was "), 72))
```

~~~text
this movie was a movie, but i hanted, a luche manet and to people ending
it was ever to do get enough. forest this movie is that it it patching
it thrily the girls they away their isney attenting for than belied fla
~~~

It is not English, but it is built like English: real words (*movie*, *people*, *ending*, *enough*,
*girls*), spaces and commas in plausible places, fragments of review phrasing ("this movie is that"),
all learned from characters alone, in about 20 seconds on a T4. On a GPU or another machine your
text will be different: compare its character, not its words. A real GPT is the same kind of model,
thousands of times larger, reading subword tokens, and trained on vastly more text.

<details><summary>(Optional) Reading the code: <code>torch.tril</code>, <code>flatten(0, 1)</code>, <code>torch.multinomial</code></summary>

**`torch.tril(torch.ones(L, L, dtype=torch.bool))`** keeps the lower triangle (row `i`, columns `0..i`)
and sets everything above the diagonal to `False`. The `(L, L)` mask broadcasts over the batch and the
heads, so one mask serves every sequence. With L21's padding mask it shares one rule: `False` scores
become minus infinity before the softmax.

**`logits.flatten(0, 1)`** merges the batch and position axes: `(64, 64, 43)` becomes `(4096, 43)`, one
row per next-character prediction, which is the shape `F.cross_entropy` expects. `y.flatten()` lines up
the 4,096 targets.

**`torch.multinomial(probs, 1, generator=g)`** draws one index at random with the given probabilities;
the seeded generator makes the draw repeatable. Dividing the logits by a **temperature** before the
softmax sharpens (below 1) or flattens (above 1) those probabilities. Exercise 3 tries it.
</details>

**Does it use order?** Step 7's classifier had positions and barely used them. Check this model the
same way it is used: give it 64 characters and score its prediction of the next one, once with the
characters in their true order, once with the first 56 shuffled (the last 8 kept in place), and once
with all 64 shuffled. The characters are the same each time; only their order changes:

```python
@torch.no_grad()
def last_char_loss(model, n_shuffled):
    """loss on the character after each validation window,
    with the window's first n_shuffled characters shuffled"""
    model.eval()
    g = torch.Generator().manual_seed(123)
    gs = torch.Generator().manual_seed(7)
    losses = []
    for _ in range(20):
        x, y = get_batch(val_ids, 128, g)
        # a random order for the first n_shuffled positions
        order = torch.rand(len(x), n_shuffled, generator=gs)
        idx = order.argsort(1).to(device)
        x[:, :n_shuffled] = x[:, :n_shuffled].gather(1, idx)
        logits = model(x, causal=True)[:, -1]
        loss = F.cross_entropy(logits, y[:, -1])
        losses.append(loss.item())
    return sum(losses) / len(losses)

for n, name in [(0, "true order"), (56, "first 56 shuffled"),
                (64, "all 64 shuffled")]:
    print(f"{name:18s} loss {last_char_loss(gpt, n):.3f}")
```

~~~text
true order         loss 1.577
first 56 shuffled  loss 1.593
all 64 shuffled    loss 4.946
~~~

Shuffle the whole window and the loss more than triples, to worse than a uniform guess (3.761): the
model reads the order of its context, not just which characters are in it. Keep the last 8 characters
in place and the loss barely moves: almost all the order it uses is in the last few characters, which
is what a small model trained for 20 seconds can be expected to learn. (Exact values vary a little
between machines; in our runs the full shuffle gave 4.83 to 4.95.)

<details><summary>(Optional) Reading the code: <code>rand().argsort(1)</code>, <code>gather</code></summary>

**`torch.rand(len(x), n_shuffled).argsort(1)`** draws random numbers for each row and returns, per row,
the positions that would sort them. Random numbers sort into a random order, so each row gets its own
random permutation of `0..n_shuffled-1`, all rows at once.

**`x[:, :n].gather(1, idx)`** builds a new tensor whose row `r`, column `j` is `x[r, idx[r, j]]`: each
row's first `n` characters are read in the shuffled order. Writing the result back into `x[:, :n]`
leaves the last `64 - n` characters where they were.
</details>

**Why the mask matters.** Train the identical model **without** the causal mask. Position `i` can then
see character `i + 1`, the very character it must predict:

```python
cheat = train_gpt(causal=False)
print(f"validation loss, as trained (sees the future):"
      f" {val_loss(cheat, causal=False):.3f}")
print(f"validation loss, future hidden:"
      f"                {val_loss(cheat, causal=True):.3f}")
print(textwrap.fill(generate(cheat, "this movie was "), 72))
```

~~~text
validation loss, as trained (sees the future): 0.035
validation loss, future hidden:                6.605
this movie was sizhvsvssssssssssssssssssssssssssssssssssssssssssssssssss
ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
sstsss ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
~~~

As trained, the unmasked model looks almost perfect (loss 0.035): position `i` reads character `i + 1`
straight from its input and copies it. Hide the future, which is exactly the situation of generation
(the next character does not exist yet), and its loss is 6.605, worse than the 3.761 of a uniform
guess. It learned nothing about text, only to copy, and its generated text collapses into one repeated
letter. The causal mask is what makes the training task the same as the task at generation time; every
decoder is trained with it.

## Your Turn

### Exercise 1 -- A learned positional encoding

Sinusoidal encodings are fixed. BERT and GPT-2/GPT-3 instead **learn** the position signal: a trainable
`nn.Embedding(max_len, d)` looked up by position, exactly like a token embedding (Step 8's `TinyGPT`
already uses one). Train a `TinyTransformer` with `pos_mode="learned"` on the order task and report its
test accuracy. Does a learned signal also solve it?

**Hint:** `torch.manual_seed(0)`, then `train(TinyTransformer(pos_mode="learned"))`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
learned positional embedding: 1.0000   (sinusoidal: 1.0000)
~~~

The learned table solves the order task as completely as the sinusoidal code: what matters is that the
positions can be told apart, not the form of the signal. BERT, GPT-2 and GPT-3 use learned positions;
most open models released since 2023 (LLaMA, Mistral, Qwen) use **rotary** encodings instead (concept
page), which add nothing to the tokens: in every attention layer they rotate the queries and keys by
their positions, so each score depends on the signed offset between the two tokens.

```python
torch.manual_seed(0)
acc_learned = train(TinyTransformer(pos_mode="learned"))
print(f"learned positional embedding: {acc_learned:.4f}"
      f"   (sinusoidal: {acc_pe:.4f})")
```
</details>

### Exercise 2 -- Rescue the deep Post-LN model with a warm-up

In Step 6 the 12-block Post-LN model failed. The 2017 paper trained its Post-LN Transformer with a
**learning-rate warm-up**: the learning rate starts near zero and rises linearly to its full value over
the first steps. Write `train_warmup`, a copy of `train` that adds a 60-step warm-up, and train the
12-block Post-LN model with it (seed 0). Does it learn now?

**Hint:** after creating `opt`, add the scheduler below, and call `sched.step()` right after every
`opt.step()`. `LambdaLR` multiplies the learning rate by the function's value at step `s`: 1/60 at the
first step, 1 from step 60 on (the whole run is 240 steps).

~~~python
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 60))
~~~

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
12 blocks, Post-LN + 60-step warm-up: 1.0000
~~~

With the warm-up, the 12-block Post-LN model learns the task perfectly instead of answering 0 for
everything. In our tests the warm-up rescued it with all three seeds, and so did a smaller constant
learning rate (0.0003): the full-size steps at the very start were the problem, not the architecture.
Here Pre-LN avoids the problem without any schedule.

```python
def train_warmup(model, warmup=60, epochs=15, bs=256, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(epoch)
        perm = torch.randperm(len(ytr), generator=g)
        model.train()
        for s in range(0, len(ytr), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss_fn(model(xtr[b]), ytr[b]).backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(xte).argmax(1)
    return (pred == yte).float().mean().item()

torch.manual_seed(0)
m = TinyTransformer(n_blocks=12)
print(f"12 blocks, Post-LN + 60-step warm-up:"
      f" {train_warmup(m):.4f}")
```
</details>

### Exercise 3 -- Sampling temperature

`generate` divides the logits by a **temperature** before the softmax. Generate 150 characters from the
trained `gpt` with the prompt `"the acting was "` at temperatures 0.5, 1.0 and 1.5 (keep `seed=0`).
How do the three texts differ?

**Hint:** `generate(gpt, "the acting was ", n=150, temperature=t)`; wrap each with
`textwrap.fill(..., 72)`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
temperature 0.5:
the acting was a movie, but i have to the should ever the seen the movie
is force that the get enough. this is produce, i better bad the talked
it the movie is so be
temperature 1.0:
the acting was a movie, bit nothing out authen't even a so in this movie
is flick who shot, but i did. fourse and anying in the praisical a
haltogical episodes doc t
temperature 1.5:
the acting was about they0 weirdnt doubbluthe 'caels. it's is thater
anwalcristle weurs-0,, youen director, be, mianying in . dap-a -tephes
havis't, obedious his you
~~~

Your three texts will differ from these on another machine; the pattern is what to compare. At 0.5
the model mostly picks its likeliest characters: the most word-like text, but repetitive ("the
should ever the seen the movie"). At 1.0 it samples from the probabilities it learned: more varied,
with more broken words. At 1.5 the distribution is flattened, unlikely characters get picked, and the
words fall apart ("weirdnt doubbluthe", stray digits). Text generators usually sample below 1, often
with further filters on top.

```python
for t in [0.5, 1.0, 1.5]:
    print(f"temperature {t}:")
    print(textwrap.fill(
        generate(gpt, "the acting was ", n=150, temperature=t),
        72))
```
</details>

### Exercise 4 -- Does a decoder need positions?

Step 5 showed that self-attention without a mask cannot see order unless it is given positions. A
decoder has a causal mask, and that changes things: position 0 sees one character, position 1 sees
two, position 63 sees all 64. Train the tiny GPT **without** its position embeddings
(`train_gpt(causal=True, pos=False)`), then compare its validation loss with the `gpt` of Step 8, run
the shuffle check `last_char_loss` on it, and generate a sample. Does it learn order?

**Hint:** `val_loss(model, causal=True)`; `last_char_loss(model, n)` for `n` in 0, 56, 64; and
`generate(model, "this movie was ")`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
validation loss, no positions: 1.591   (Step 8, learned positions: 1.544)
true order         loss 1.642
first 56 shuffled  loss 1.661
all 64 shuffled    loss 5.047
this movie was a movie, but i hand to love he maned and horrorible movie
in fille who shot, but i did. foor because tire in the producerst budget
to the stufffic the startest fair innemarks up her actor but pay punk
~~~

Yes. Without any position signal the decoder ends only a little behind the one with learned positions
(1.591 against 1.544; in our runs on 1, 2 and the default number of CPU threads the gap was always
about 0.05), writes
text of the same kind, and shuffling its context hurts it just as much: it uses order. The reason is
the causal mask. Without a mask, self-attention treats its input as a set, which is why Step 5's
encoder needed positions. With the mask, position 0 sees one character, position 1 sees two, position
63 sees 64, so a position can work out from what it may see roughly where it is. Haviv et al. (2022)
found the same in much larger language models. Positions still help a little here, and real decoders
use them; but "a Transformer without positional encoding is blind to order" is true only of attention
without a causal mask.

```python
nopos = train_gpt(causal=True, pos=False)
print(f"validation loss, no positions: "
      f"{val_loss(nopos, causal=True):.3f}"
      f"   (Step 8, learned positions:"
      f" {val_loss(gpt, causal=True):.3f})")
for n, name in [(0, "true order"), (56, "first 56 shuffled"),
                (64, "all 64 shuffled")]:
    print(f"{name:18s} loss {last_char_loss(nopos, n):.3f}")
print(textwrap.fill(generate(nopos, "this movie was "), 72))
```
</details>

## Summary

- **Multi-head attention** runs `h` attentions side by side; each head projects the whole token to
  `d/h` numbers, and the results are concatenated and mixed by `Wo`. It matched `nn.MultiheadAttention`
  to within `1e-6`. Its matrix products cost the same as one head's, but the softmax runs `h` times, and
  stored weights take `h` matrices of `L x L`.
- **Positional encoding** adds a position signal to every token. Without it, the encoder's order task
  stayed at 0.4810, no better than always answering 1 (0.517), and the logits did not change when the
  tokens after `CLS` were shuffled; with it, 1.0000.
- **Attention weights are not an explanation.** The last block's heads looked specialized (0.998 on A,
  0.972 on B), but the ablation showed the B head could go at no cost, while head 3, whose A and B
  averages looked mixed, mattered most (0.9420 without it): it looks at whichever marker comes first.
- **Layer norm** standardizes each token over its own features, the same alone or in any batch, unlike
  batch norm. The **encoder block** wraps self-attention and a feed-forward network in residual + layer
  norm; it matched `nn.TransformerEncoderLayer` and has 33,472 parameters at `d = 64`.
- **Depth:** without the residual, 6 blocks were stuck at 0.4830 (answering 0 every time); with it
  they trained. At 12 blocks, Post-LN was stuck without a warm-up (Exercise 2 rescues it), while Pre-LN
  reached 1.0000.
- **IMDB:** positional encoding made the same-word sentences score differently, but the more positive
  one scored higher in only 6 of 8 pairs and accuracy did not move (0.8483 both): under these
  conditions the model has position available and, most likely, too little reason to use it. A
  **language model** (the tiny GPT) gets a target at every position and needs order to meet it; with the
  **causal mask** it learned to write review-like text and to use the order of what it reads (shuffling
  its context sent its loss from 1.577 to 4.946), and without the mask it "learned" to copy the next
  character and could not generate at all.
- **A causal decoder learns order without positions** (Exercise 4): the mask tells each position how
  many tokens precede it. "Blind to order" is a property of attention without a causal mask.
