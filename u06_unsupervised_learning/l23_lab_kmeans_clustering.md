---
title: "Lab: Finding Groups Without Labels"
unit: "VI"
lesson: "23"
type: lab
tags: [unsupervised-learning, clustering, k-means, silhouette, semi-supervised, scikit-learn]
difficulty: intermediate
duration: "60 mins"
---

**Goal:** take a pile of points with **no labels** and let K-Means find the groups. In Part A
you fit K-Means, see its regions, run its loop **by hand**, watch it get stuck from a bad start,
pick the number of clusters with the elbow and the silhouette, and then find where it breaks.
In Part B you put it to work on real data: shrinking an image to a handful of colors, and
choosing which 50 handwritten digits are worth labeling. Pairs with the concept note
[K-Means Clustering & Similarity](l23_concept_kmeans_clustering.qmd).

> **Previously:** L22 -- The Transformer Architecture (closes Unit V)  |  **Next:** L24 -- Hierarchical & Density-Based Clustering

> This page is the read-only view. To run the lab, open the notebook (`l23_lab_kmeans_clustering.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u06_unsupervised_learning/l23_lab_kmeans_clustering.ipynb)

## Scenario

We have **1500 points in 2-D and no answer key** -- no class, no label, nothing but the
coordinates. Our job is the unsupervised one: find the structure that is already in the data.
The points happen to fall in four roughly round blobs, but K-Means is never told that; it has
to discover the groups from distance alone. We keep the true labels off to the side for one
purpose only -- to *score* the clustering -- and the algorithm never sees them.

The data in Part A is **synthetic** (a fixed seed, identical for everyone); two dimensions keep
every step plottable. Part B uses two small datasets that ship with scikit-learn (a photo and
the 8x8 handwritten digits), so nothing is downloaded.

## Setup

The setup is **two cells** (the pattern every lab uses). The first only installs; the second
imports, seeds, and builds the data.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships scikit-learn, numpy and matplotlib, so this is effectively a no-op there.
%pip install -q scikit-learn
# local, in a terminal (not in the notebook):  uv add scikit-learn numpy matplotlib
```

```python
# Setup, cell 2 of 2 -- IMPORTS, SEED, DATA (safe to re-run without re-installing)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import (make_blobs, make_moons,
                              load_sample_image, load_digits)
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             adjusted_rand_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Four roughly round, well-separated blobs in 2-D. We keep y_true ONLY to score the
# clustering -- K-Means never sees it.
X, y_true = make_blobs(n_samples=1500, centers=4, cluster_std=0.70, random_state=0)
print(X.shape[0], "points,", X.shape[1], "features, and no labels for clustering")
```

~~~text
1500 points, 2 features, and no labels for clustering
~~~

## Part A -- How K-Means Works

### Step 1: Cluster It -- No Labels Required

K-Means asks one thing up front: **how many clusters, `k`?** Say four for now (we will learn
to choose `k` in Step 4). `fit_predict` runs the algorithm and hands back a cluster label for
every point:

```python
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

print("first 10 cluster labels:", y_pred[:10])
print("points per cluster:    ", np.bincount(y_pred))
print("centroids (rounded):")
print(kmeans.cluster_centers_.round(2))
```

~~~text
first 10 cluster labels: [0 0 0 2 1 0 3 2 0 0]
points per cluster:     [373 375 375 377]
centroids (rounded):
[[ 0.89  4.29]
 [-1.28  7.8 ]
 [-1.58  2.93]
 [ 2.1   0.9 ]]
~~~

K-Means split the 1500 points into four nearly equal groups and reported a **centroid** (the
mean point) for each. Every point belongs to its **nearest** centroid, so the centroids carve the
whole plane into regions -- one per centroid, called **Voronoi cells**. Any new point that lands
in a cell gets that cell's label. The helper below paints the cells by asking the model to
`predict` a fine grid of points:

```python
def plot_cells(model, X, ax, title):
    # Color every spot of the plane by its nearest centroid.
    lo, hi = X.min(axis=0) - 1, X.max(axis=0) + 1
    g0, g1 = np.meshgrid(np.linspace(lo[0], hi[0], 300),
                         np.linspace(lo[1], hi[1], 300))
    grid = np.column_stack([g0.ravel(), g1.ravel()])
    cells = model.predict(grid).reshape(g0.shape)
    ax.pcolormesh(g0, g1, cells, cmap="tab10", vmin=0, vmax=9,
                  alpha=0.2, shading="auto")
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=5,
               cmap="tab10", vmin=0, vmax=9)
    c = model.cluster_centers_
    ax.scatter(c[:, 0], c[:, 1], marker="X", s=180, c="black")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

fig, ax = plt.subplots(figsize=(7, 4.5))
plot_cells(kmeans, X, ax, "K-Means (k=4): points, centroids and cells")
plt.show()
```

You should see four blobs, each in its own color, a black **X** on each centroid, and the plane
behind them split into four tinted regions. Every border between two regions is a **straight
line** -- the set of spots equally far from two centroids. Remember that: it is the root of the
failures in Step 5.

One more thing to internalize now: the labels `0,1,2,3` are **arbitrary integers**, not
classes. Fit the same model with a few different seeds and compare. To check whether two
clusterings form the **same groups** regardless of their numbering, we use the **Adjusted Rand
Index** (ARI): 1.0 means the two groupings agree exactly, about 0 means no better than chance,
and renaming the clusters does not change it.

```python
for seed in (0, 1, 2):
    km_s = KMeans(n_clusters=4, random_state=seed).fit(X)
    x1_of_label = km_s.cluster_centers_[:, 0].round(1).tolist()
    same = adjusted_rand_score(y_pred, km_s.labels_)
    print(f"seed {seed}: centroid x1 for labels 0-3 = {x1_of_label}, "
          f"ARI vs ours {same:.3f}")
```

~~~text
seed 0: centroid x1 for labels 0-3 = [2.1, -1.3, -1.6, 0.9], ARI vs ours 0.998
seed 1: centroid x1 for labels 0-3 = [2.1, -1.3, -1.6, 0.9], ARI vs ours 0.993
seed 2: centroid x1 for labels 0-3 = [0.9, -1.3, 2.1, -1.6], ARI vs ours 0.993
~~~

The blob centered at x1 = 2.1 is cluster `0` for seeds 0 and 1 but cluster `2` for seed 2 (and
cluster `3` in our own run above). Yet the ARI against our clustering is 0.99 or more every
time: the **same four groups**, just numbered differently, with a handful of boundary points
swapping sides. Labels name groups; they carry no meaning of their own.

### Step 2: Similarity Is Distance -- and the Loop by Hand

K-Means has exactly one notion of "similar": **close in Euclidean distance to a centroid**.
`transform` makes that explicit -- it returns each point's distance to all four centroids. Try
three brand-new points:

```python
X_new = np.array([[0.0, 4.0], [-1.0, 8.0], [2.0, 1.0]])
dist = kmeans.transform(X_new)
print("distance to each of the 4 centroids (rounded):")
print(dist.round(2))
print("nearest-centroid label:", kmeans.predict(X_new))
print("affinity 1/(1+distance):")
print((1 / (1 + dist)).round(2))
```

~~~text
distance to each of the 4 centroids (rounded):
[[0.94 4.01 1.91 3.74]
 [4.16 0.35 5.1  7.75]
 [3.47 7.55 4.07 0.14]]
nearest-centroid label: [0 1 3]
affinity 1/(1+distance):
[[0.52 0.2  0.34 0.21]
 [0.19 0.74 0.16 0.11]
 [0.22 0.12 0.2  0.88]]
~~~

Each new point is assigned to the centroid it is **closest** to (smallest distance in its row).
Distance runs from 0 to infinity, which is awkward to compare, so we often turn it into an
**affinity** with `1/(1+distance)`: 1 when you are sitting on the centroid, fading toward 0 as
you move away. The third point sits almost on top of centroid 3 (distance 0.14, affinity 0.88).

That "assign to the nearest centroid" rule is just an `argmin` over the distance rows. We can
reproduce K-Means' own assignments by hand:

```python
byhand = kmeans.transform(X).argmin(axis=1)
print("assign-to-nearest by hand == sklearn labels_?", np.all(byhand == kmeans.labels_))
```

~~~text
assign-to-nearest by hand == sklearn labels_? True
~~~

That is half of the K-Means loop. The other half is **recentering**: move each centroid to the
mean of the points assigned to it. Let us run the whole loop -- Lloyd's algorithm -- ourselves,
in plain NumPy, from a deliberately poor start: four centroids in a row **below** all the data.

```python
def assign(X, C):
    # squared distance from every point to every centroid
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)

def recenter(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])

C = np.array([[-3.0, -1.0], [-1.0, -1.0], [1.0, -1.0], [3.0, -1.0]])
path = [C]
for it in range(1, 20):
    labels = assign(X, C)
    inertia = ((X - C[labels]) ** 2).sum()
    print(f"iteration {it}: inertia {inertia:8.1f}")
    C_next = recenter(X, labels, 4)
    if np.allclose(C_next, C):
        print("centroids stopped moving -- converged")
        break
    C = C_next
    path.append(C)

print(f"by hand: {inertia:.1f}   sklearn: {kmeans.inertia_:.1f}")
```

~~~text
iteration 1: inertia  47828.5
iteration 2: inertia   4733.8
iteration 3: inertia   1569.4
iteration 4: inertia   1357.8
iteration 5: inertia   1352.2
iteration 6: inertia   1352.2
iteration 7: inertia   1352.2
centroids stopped moving -- converged
by hand: 1352.2   sklearn: 1352.2
~~~

Two short functions and a loop reproduce K-Means. **Inertia** -- the sum of squared distances
from each point to its own centroid -- falls at every pass (47828 -> 4734 -> 1569 -> ...) and
then stops changing: neither step can ever raise it, and there are only finitely many ways to
split 1500 points, so the loop must stop. It lands on the same inertia scikit-learn found. Plot
the centroids' journey:

```python
path = np.array(path)
plt.figure(figsize=(7, 4.5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap="tab10", vmin=0, vmax=9)
for j in range(4):
    plt.plot(path[:, j, 0], path[:, j, 1], "k.-", lw=1.5, ms=9)
plt.scatter(path[0, :, 0], path[0, :, 1], marker="s", s=80,
            facecolors="white", edgecolors="black", label="start")
plt.scatter(path[-1, :, 0], path[-1, :, 1], marker="X", s=180,
            c="black", label="converged")
plt.legend(loc="upper right")
plt.title("Lloyd's algorithm by hand: each centroid's path")
plt.xlabel("x1"); plt.ylabel("x2"); plt.show()
```

The four white squares start in a row below the data. After the first pass each centroid jumps
toward the mean of whatever points were nearest to it, and within a few passes each one has
climbed into the middle of its own blob. The big jumps come first; the last moves are tiny.

### Step 3: Local Optima -- Why the Start Matters

The loop always converges, but **not always to the best answer**. Lloyd only ever goes
downhill, so where it ends depends on where it starts. Start it from 4 random data points
(`init="random"`), run it once (`n_init=1`), and repeat with 50 different seeds. A good run
reaches inertia about 1352; count how many get stuck above 1400:

```python
def count_stuck(**options):
    stuck = 0
    for seed in range(50):
        km = KMeans(n_clusters=4, random_state=seed, **options).fit(X)
        stuck += km.inertia_ > 1400
    return stuck

print("random start,  1 run :", count_stuck(init="random", n_init=1), "of 50 stuck")
print("k-means++,     1 run :", count_stuck(init="k-means++", n_init=1), "of 50 stuck")
print("random start, 10 runs:", count_stuck(init="random", n_init=10), "of 50 stuck")
print("default n_init:", KMeans().n_init)
```

~~~text
random start,  1 run : 12 of 50 stuck
k-means++,     1 run : 2 of 50 stuck
random start, 10 runs: 0 of 50 stuck
default n_init: auto
~~~

A random start gets stuck about one time in four. **k-means++** -- which picks starting
centroids spread far apart -- cuts that to 2 in 50. Running the whole algorithm **10 times**
and keeping the lowest-inertia result (`n_init=10`) removed it entirely here. Note the default:
`n_init="auto"` means **a single run** when the start is k-means++ (scikit-learn 1.4 and later),
so the safety net of several runs is something **you** switch on when it matters. Here is what
a stuck run looks like:

```python
stuck_km = KMeans(n_clusters=4, init="random", n_init=1, random_state=1).fit(X)
print("inertia:", round(stuck_km.inertia_, 1),
      "| cluster sizes:", np.bincount(stuck_km.labels_),
      "| ARI vs truth:", round(adjusted_rand_score(y_true, stuck_km.labels_), 3))

fig, ax = plt.subplots(figsize=(7, 4.5))
plot_cells(stuck_km, X, ax, "Stuck in a local optimum (random start, seed 1)")
plt.show()
```

~~~text
inertia: 2719.8 | cluster sizes: [382 171 740 207] | ARI vs truth: 0.612
~~~

The loop converged -- but to a bad split. Two true blobs share one centroid (the cluster of
740 points) while another blob is cut in two (171 + 207). Neither Lloyd step can fix it: moving
any single centroid a little only makes inertia worse, so the loop is trapped. That is a
**local optimum**, and it is why the start and the number of runs matter.

### Step 4: How Many Clusters?

We told K-Means `k=4`, but on real data you do not know `k`. The obvious idea -- pick the `k`
with the lowest inertia -- does not work. Watch inertia as `k` grows:

```python
inertias = [KMeans(n_clusters=kk, random_state=42).fit(X).inertia_ for kk in range(1, 10)]
for kk, val in zip(range(1, 10), inertias):
    print(f"  k={kk}: inertia {val:8.1f}")
```

~~~text
  k=1: inertia  14370.7
  k=2: inertia   6162.9
  k=3: inertia   2831.6
  k=4: inertia   1352.2
  k=5: inertia   1227.4
  k=6: inertia   1115.8
  k=7: inertia    993.2
  k=8: inertia    896.6
  k=9: inertia    789.2
~~~

Inertia keeps falling as `k` rises -- more centroids sit closer to the points, and at `k = n`
(one centroid per point) it hits zero. So you cannot pick `k` by minimizing it. (Strictly, it is
the **best possible** inertia that can only fall; a single Lloyd run could land in a local
optimum and break the pattern, but with a good start it holds, as here.) Look at *how* it falls:
a huge drop through `k=4` (14371 -> 1352), then a near-flat crawl. That bend is the **elbow**:

```python
plt.figure(figsize=(7, 4))
plt.plot(range(1, 10), inertias, "o-")
plt.xlabel("k (number of clusters)"); plt.ylabel("inertia")
plt.title("Elbow: inertia vs k"); plt.grid(True); plt.show()
```

The curve plunges and then flattens, with the kink at `k=4`: extra clusters past that point buy
very little. The elbow is a judgment call by eye. The **silhouette score** makes it
quantitative: for each point it compares the mean distance to the other points of its own
cluster (`a`) with the mean distance to the points of the nearest other cluster (`b`), as
`(b - a) / max(a, b)` -- from -1 (wrong cluster) to +1 (deep inside the right one). The mean over
all points is the silhouette score, and the best `k` is the one that **maximizes** it. Complete
the sweep:

```python
# Uncomment and complete the marked line, then run:
# sil = [silhouette_score(X, KMeans(n_clusters=____, random_state=42).fit(X).labels_)  # cluster count = the loop variable
#        for kk in range(2, 10)]
# for kk, s in zip(range(2, 10), sil):
#     print(f"  k={kk}: silhouette {s:.3f}")
# print("best k (max silhouette):", list(range(2, 10))[int(np.argmax(sil))])
```

<details><summary>Expected Output</summary>

~~~text
  k=2: silhouette 0.538
  k=3: silhouette 0.573
  k=4: silhouette 0.627
  k=5: silhouette 0.536
  k=6: silhouette 0.465
  k=7: silhouette 0.428
  k=8: silhouette 0.308
  k=9: silhouette 0.329
~~~

~~~text
best k (max silhouette): 4
~~~

The silhouette score rises to a clear peak at **k=4 (0.627)**, then declines -- it recovered the
true number of blobs (the far tail wobbles a little, but k=4 is the unambiguous maximum). The
blank is `n_clusters=kk`: each pass clusters with the current loop value `kk` and scores the
result. The silhouette costs more than inertia -- every point is compared with every other
point -- but, unlike inertia, it does not simply reward more clusters.
</details>

A single average hides a lot. The **silhouette plot** shows every point's coefficient. Take two
overlapping blobs -- one tight, one loose -- cluster them with `k=2`, and draw the points next to
their silhouettes. Each cluster's coefficients are sorted into a horizontal band of thin bars,
one bar per point -- so a band's **height** is the cluster's size, and how far **right** it
reaches is how well its points are placed. Points with `s < 0` (closer to the *other* cluster
than to their own) are circled in red:

```python
def silhouette_panels(X, labels, ax_pts, ax_sil, title):
    s = silhouette_samples(X, labels)
    ax_pts.scatter(X[:, 0], X[:, 1], c=labels, s=8, cmap="tab10", vmin=0, vmax=9)
    neg = s < 0
    ax_pts.scatter(X[neg, 0], X[neg, 1], s=60, facecolors="none",
                   edgecolors="red", linewidths=1.4, label="s < 0")
    ax_pts.legend(loc="lower left")
    ax_pts.set_title(title)
    bottom = 0
    for c in np.unique(labels):
        sc = np.sort(s[labels == c])
        ax_sil.barh(np.arange(bottom, bottom + len(sc)), sc, height=1.0,
                    color=plt.colormaps["tab10"](c))
        ax_sil.text(-0.05, bottom + len(sc) / 2, f"cluster {c}", ha="right")
        bottom += len(sc) + 15
    ax_sil.axvline(s.mean(), color="black", linestyle="--",
                   label=f"mean = {s.mean():.3f}")
    ax_sil.axvline(0, color="gray", linewidth=0.8)
    ax_sil.set_xlim(-0.5, 1.0)
    ax_sil.set_yticks([])
    ax_sil.set_xlabel("silhouette coefficient s")
    ax_sil.legend(loc="lower right")
    return s

X2, _ = make_blobs(n_samples=300, centers=[[-1.5, 0.0], [1.5, 0.5]],
                   cluster_std=[0.8, 1.2], random_state=3)
km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X2)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))
s2 = silhouette_panels(X2, km2.labels_, ax_a, ax_b, "Two overlapping blobs, k=2")
plt.show()

print("silhouette score:", round(s2.mean(), 3))
for c in (0, 1):
    in_c = km2.labels_ == c
    print(f"cluster {c}: {in_c.sum()} points, mean s {s2[in_c].mean():.3f}, "
          f"{np.sum(s2[in_c] < 0)} below 0")
```

~~~text
silhouette score: 0.487
cluster 0: 134 points, mean s 0.395, 5 below 0
cluster 1: 166 points, mean s 0.561, 0 below 0
~~~

Read the plot band by band. The **tight** cluster (1) is the taller band (166 points) and reaches
0.7: most of its points sit deep inside it (mean s 0.561). The **loose** cluster (0) is shorter
(134 points), stops near 0.6 (mean 0.395), and has a tail that crosses 0 -- the five red-circled points on the scatter, all on the side
facing the other cluster. Points near the shared border score near 0 in both clusters. The one
number 0.487 tells you "decent"; the plot tells you **which** points are well placed and which
are not.

### Step 5: Where K-Means Breaks

K-Means' one tool -- distance to a centroid, with straight borders between cells -- bakes in an
assumption: clusters are **round, compact blobs**. When the real groups are not shaped like
that, K-Means fails. The classic example is two interleaved crescents (`make_moons`). Run the
same algorithm and score it against the true crescents with the ARI:

```python
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
km_moons = KMeans(n_clusters=2, random_state=42).fit(X_moons)

print("MOONS  silhouette:           ", round(silhouette_score(X_moons, km_moons.labels_), 3))
print("MOONS  ARI vs true crescents:", round(adjusted_rand_score(y_moons, km_moons.labels_), 3))
print("BLOBS  ARI vs truth:         ", round(adjusted_rand_score(y_true, y_pred), 3))
```

~~~text
MOONS  silhouette:            0.489
MOONS  ARI vs true crescents: 0.239
BLOBS  ARI vs truth:          0.962
~~~

On the blobs, K-Means matched the truth almost perfectly (**ARI 0.962**). On the moons it drops
to **ARI 0.239**. That does not mean the result is random -- count the points:

```python
match = np.mean(km_moons.labels_ == y_moons)
print("points on the right side:", round(max(match, 1 - match), 3))
```

~~~text
points on the right side: 0.745
~~~

About three points in four land with their own crescent. The failure is **structural**: one
straight border cuts across **both** crescents, so a big chunk of each is handed to the other
group every time. Now the surprise. The silhouette of K-Means' wrong split was 0.489 -- a
respectable score. What does the **true** grouping score?

```python
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
s_km = silhouette_panels(X_moons, km_moons.labels_, axes[0, 0], axes[0, 1],
                         "K-Means split (wrong)")
s_true = silhouette_panels(X_moons, y_moons, axes[1, 0], axes[1, 1],
                           "True crescents (right)")
plt.tight_layout(); plt.show()

print("silhouette, K-Means split:  ", round(s_km.mean(), 3))
print("silhouette, true crescents: ", round(s_true.mean(), 3))
```

~~~text
silhouette, K-Means split:   0.489
silhouette, true crescents:  0.334
~~~

The silhouette **prefers the wrong answer** (0.489 against 0.334). The true crescents have
red-circled points at both tips, which curl closer to the other crescent than to the far end of
their own. The silhouette measures "close to my cluster, far from the other", the same compact,
round notion of a cluster K-Means uses -- so it cannot catch this mistake. Treat it as a guide
for compact clusters, never as a truth meter.

Crescents are not the only trouble. Two more shapes from scikit-learn's own examples: blobs
**stretched** along a slant, and blobs with **very different spreads**:

```python
Xb, yb = make_blobs(n_samples=1500, random_state=170)
stretch = np.array([[0.6, -0.64], [-0.41, 0.85]])
X_str = Xb @ stretch
X_spr, y_spr = make_blobs(n_samples=1500, cluster_std=[1.0, 2.5, 0.5],
                          random_state=170)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (name, Xs, ys) in zip(axes, [("stretched", X_str, yb),
                                     ("unequal spread", X_spr, y_spr)]):
    km_s = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
    ari = adjusted_rand_score(ys, km_s.labels_)
    print(f"{name:15s} ARI vs truth: {ari:.3f}")
    plot_cells(km_s, Xs, ax, f"{name} (ARI {ari:.3f})")
plt.tight_layout(); plt.show()
```

~~~text
stretched       ARI vs truth: 0.599
unequal spread  ARI vs truth: 0.826
~~~

On the **stretched** blobs K-Means cuts straight across the two parallel stripes on the left, so
each of those clusters gets the upper or lower piece of **both** stripes (ARI 0.599): a cell
built around a center cannot follow a long, slanted group. With **unequal spreads** (ARI 0.826) the border
sits halfway between centroids, so the wide cluster's outer points spill into its tight
neighbors' cells. Both are the same blind spot as the crescents: K-Means knows centers and
distances, not shapes. That is the opening for **L24**, where DBSCAN clusters by connected
density and hierarchical clustering merges by linkage -- both follow shapes K-Means cuts in two.

## Part B -- Clustering at Work

### Step 6: Color Quantization

A color photo is a cloud of points too: every pixel is a point in 3-D **color space** (red,
green, blue, each between 0 and 1). Cluster those points with `k` centroids, then repaint each
pixel with its centroid's color, and the photo uses only `k` colors. That is **color
quantization** -- the idea behind GIF palettes and poster effects.

*Data provenance:* `china.jpg` ships with scikit-learn (`load_sample_image`); photo by
danielbuechele on Flickr, CC BY 2.0.

```python
china = load_sample_image("china.jpg") / 255.0
pixels = china.reshape(-1, 3)
n_colors = len(np.unique(pixels, axis=0))
print(china.shape, "image ->", len(pixels), "pixels,", n_colors, "distinct colors")
```

~~~text
(427, 640, 3) image -> 273280 pixels, 96615 distinct colors
~~~

Fitting on every 10th pixel is plenty to place the centroids; then every pixel is assigned to
its nearest one:

```python
sample = pixels[::10]

def show(ax, img, title):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

fig, axes = plt.subplots(1, 5, figsize=(16, 2.8))
show(axes[0], china, "original")
for ax, kk in zip(axes[1:], (2, 4, 8, 16)):
    km_c = KMeans(n_clusters=kk, random_state=42).fit(sample)
    palette = km_c.cluster_centers_
    repainted = palette[km_c.predict(pixels)].reshape(china.shape)
    err = np.mean((repainted - china) ** 2)
    print(f"k={kk:2d} colors: mean squared color error {err:.4f}")
    show(ax, repainted, f"k={kk} colors")
plt.show()
```

~~~text
k= 2 colors: mean squared color error 0.0198
k= 4 colors: mean squared color error 0.0070
k= 8 colors: mean squared color error 0.0033
k=16 colors: mean squared color error 0.0018
~~~

With 2 colors the photo is a two-tone outline; with 4 the temple, trees and sky separate; by 8-16 it
is hard to tell from the original at this size, though it uses 16 colors instead of 96,615. The
error is the image version of inertia, and it falls with `k` for the same reason. Here `k` is
not "the true number of groups" -- there is none -- but a **budget** you choose.

### Step 7: Which Images Should a Human Label?

Labels are expensive: someone has to look at each example. Clustering can tell you **which**
examples are worth labeling. The 8x8 handwritten digits (1797 images, 64 pixel features) make
the test. We hold out a quarter as a test set and pretend the 1347 training images are
**unlabeled**.

*Data provenance:* `load_digits` ships with scikit-learn; it is the UCI "Optical Recognition of
Handwritten Digits" test set (Alpaydin and Kaynak, 1998), 1797 images from 13 writers.

```python
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(
    digits.data, digits.target, test_size=0.25,
    stratify=digits.target, random_state=42)
print("train:", X_tr.shape, " test:", X_te.shape)

def score_with(X_lab, y_lab):
    # train a classifier on the labeled images, report test accuracy
    clf = LogisticRegression(max_iter=5000).fit(X_lab, y_lab)
    return clf.score(X_te, y_te)

print("all 1347 labels      :", round(score_with(X_tr, y_tr), 3))
```

~~~text
train: (1347, 64)  test: (450, 64)
all 1347 labels      : 0.962
~~~

With every label, a logistic regression (L04) reaches 0.962. Suppose the budget is **50
labels**. The naive plan labels 50 random images. Since luck matters, try 20 different random
draws:

```python
rng = np.random.RandomState(42)
random_accs = []
for _ in range(20):
    idx = rng.choice(len(X_tr), size=50, replace=False)
    random_accs.append(score_with(X_tr[idx], y_tr[idx]))
print(f"50 random labels     : mean {np.mean(random_accs):.3f} "
      f"(worst {min(random_accs):.3f}, best {max(random_accs):.3f})")
```

~~~text
50 random labels     : mean 0.831 (worst 0.736, best 0.887)
~~~

The clustering plan: run K-Means with **k=50** on the unlabeled images, and label only the image
**closest to each centroid** -- the most typical member of each cluster:

```python
km50 = KMeans(n_clusters=50, n_init=10, random_state=42).fit(X_tr)
rep_idx = km50.transform(X_tr).argmin(axis=0)
# In real life a person would label these 50 images now:
y_rep = y_tr[rep_idx]
print("digits covered by the 50:", len(np.unique(y_rep)), "of 10")
rep_acc = score_with(X_tr[rep_idx], y_rep)
print(f"50 representative labels: {rep_acc:.3f}")
```

~~~text
digits covered by the 50: 10 of 10
50 representative labels: 0.900
~~~

Same budget, better result: **0.900** against a random-draw average of 0.831, and better than
the luckiest of the 20 random draws (0.887). The 50 representatives cover all ten digits and the
different ways each one is written, while a random draw over-samples some styles and misses
others. Here are the images the person would label:

```python
fig, axes = plt.subplots(5, 10, figsize=(10, 5.6))
for ax, i in zip(axes.ravel(), rep_idx):
    ax.imshow(X_tr[i].reshape(8, 8), cmap="gray_r")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(str(y_tr[i]), fontsize=9)
plt.suptitle("The 50 representative images (nearest each centroid)")
plt.show()
```

Each tile is one cluster's most typical image, titled with its label. Every digit appears, and
most appear several times in different handwriting (look at the 1s, 2s and 7s), because K-Means
gave each common writing style its own cluster. This is **semi-supervised learning**: a little labeled data, a
lot of unlabeled data, and clustering to connect them.

## Your Turn

### Exercise 1 -- Always scale first

K-Means measures distance, so a feature on a larger numeric scale dominates the result. Here
are four groups arranged in a 2x2 grid, but feature 1 is on a scale about **1000x** larger than
feature 0. Cluster the data **as-is**, then again after `StandardScaler`, and score each against
the true groups with ARI.

```python
rng = np.random.RandomState(42)
n = 250
rows, groups, lab = [], [], 0
for cs in (0.0, 1.0):              # feature 0: small scale
    for cb in (0.0, 1000.0):       # feature 1: large scale (about 1000x)
        rows.append(np.column_stack([rng.normal(cs, 0.10, n), rng.normal(cb, 100.0, n)]))
        groups.append(np.full(n, lab)); lab += 1
Xg = np.vstack(rows); yg = np.concatenate(groups)
print("Xg shape:", Xg.shape)
```

~~~text
Xg shape: (1000, 2)
~~~

**Hint:** fit `KMeans(n_clusters=4, random_state=42)` on `Xg`, then on `StandardScaler().fit_transform(Xg)`; print `adjusted_rand_score(yg, labels_)` for each.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
ARI unscaled: 0.334
ARI scaled:   1.0
~~~

Unscaled, the big-scale feature 1 swamps the distance: K-Means can only "see" that axis, finds
the two-way split along it, and butchers the rest -- **ARI 0.334**. After `StandardScaler` gives
both features equal spread, all four groups separate cleanly -- **ARI 1.0**. Scaling is not
optional for K-Means; it is part of the recipe (the same standardization you used in L08/L10).
</details>

### Exercise 2 -- Does spreading the labels help?

Step 7 labeled 50 representatives and trained on those 50 images only. But each representative
speaks for a whole cluster. **Label propagation** copies each representative's label to every
image in its cluster, giving 1347 (imperfect) labels for the price of 50. Try two versions:

1. Propagate to **every** image: `y_prop = y_rep[km50.labels_]`. Print how many propagated labels
   are correct (compare with `y_tr`, which in real life you would not have), then train on all
   1347 images with `score_with(X_tr, y_prop)`.
2. Propagate only to the **20% of each cluster closest to its centroid** -- the safest guesses.
   `km50.transform(X_tr).min(axis=1)` is each image's distance to its own centroid; inside each
   cluster keep the images at or below the 20th percentile (`np.percentile`).

Compare both with the 0.900 of the 50 representatives alone.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
propagated labels correct  : 0.927
propagate to all 1347      : 0.891
closest 20% (291 images)   : 0.900
~~~

Propagation gave **no gain here**: 0.891 with every image, 0.900 with the closest 20%, against
0.900 for the 50 representatives alone. Spreading to all images adds about 7% wrong labels
(0.927 correct), and those mistakes cost as much as the extra data helps; the closest 20% are
almost all correct but add little the representatives did not already show. It is not a rule:
in a separate run on 10,000 of the larger 28x28 MNIST digits (not part of this lab; 100
representatives), propagation to
every image lifted test accuracy from 0.809 to 0.837. Whether it pays depends on how pure the
clusters are -- which is why you measure it, as here. One possible solution:

~~~python
y_prop = y_rep[km50.labels_]
print("propagated labels correct  :", round(np.mean(y_prop == y_tr), 3))
print("propagate to all 1347      :", round(score_with(X_tr, y_prop), 3))

d_own = km50.transform(X_tr).min(axis=1)
keep = np.zeros(len(X_tr), dtype=bool)
for c in range(50):
    in_c = km50.labels_ == c
    keep |= in_c & (d_own <= np.percentile(d_own[in_c], 20))
acc = score_with(X_tr[keep], y_prop[keep])
print(f"closest 20% ({keep.sum()} images)   : {acc:.3f}")
~~~
</details>

## Summary

- K-Means clusters **unlabeled** points by distance to `k` **centroids**; each centroid owns a
  **Voronoi cell** with straight borders. The labels are arbitrary integers -- compare
  clusterings with the **ARI**, which ignores the numbering.
- **Lloyd's algorithm** alternates assign-to-nearest (`argmin`) and recenter-on-the-mean; by hand
  it fell 47828 -> 1352.2 and matched scikit-learn exactly.
- It converges to a **local** optimum: random starts got stuck 12 times in 50, k-means++ 2, and
  `n_init=10` none. The default `n_init="auto"` is **one** run with k-means++.
- Inertia only falls with `k`, so use the **elbow** (k=4) and the **silhouette** (peak 0.627 at
  k=4); the **silhouette plot** shows which points are well placed (s near 1) and which are not
  (s < 0).
- K-Means assumes round, compact clusters: it **fails on crescents** (ARI 0.239, though 74.5% of
  points land right), stretched blobs (0.599) and unequal spreads (0.826) -- and the silhouette
  shares the bias, preferring the wrong moons split (0.489) to the truth (0.334).
- At work: **color quantization** (16 colors for 96,615) and **choosing what to label** (50
  representative digits: 0.900 vs 0.831 for random). Always **scale first** (Ex1: 0.334 -> 1.0).
- Next (L24): hierarchical and density-based clustering -- DBSCAN and agglomerative -- which
  handle the shapes K-Means cannot.
