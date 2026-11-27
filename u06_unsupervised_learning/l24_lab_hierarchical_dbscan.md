---
title: "Lab: Shapes, Trees, and Crime Hotspots"
unit: "VI"
lesson: "24"
type: lab
tags: [unsupervised-learning, dbscan, hdbscan, hierarchical-clustering, dendrogram, spatial-data, scikit-learn]
difficulty: intermediate
duration: "70 mins"
---

**Goal:** pick up exactly where L23 left off -- K-Means drawing one straight border across two
crescents -- and fix it two ways. In Part A you run **DBSCAN** (density-based) on the same moons,
see its core / border / noise points, find the range of `eps` that works, and build a
**dendrogram** whose **linkage** decides whether the crescents survive. In Part B you take
density clustering to real data: where do robberies concentrate in Puerto Rico? You will watch
hotspots appear and merge as you turn DBSCAN's dial, let **HDBSCAN** handle a city and a small town
at once, and see why hierarchical clustering is the wrong tool for this job. Pairs with the concept
note [Hierarchical & Density-Based Clustering](l24_concept_hierarchical_dbscan.qmd).

> **Previously:** L23 -- K-Means Clustering & Similarity  |  **Next:** L25 -- Gaussian Mixtures & EM

> This page is the read-only view. To run the lab, open the notebook (`l24_lab_hierarchical_dbscan.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u06_unsupervised_learning/l24_lab_hierarchical_dbscan.ipynb)

## Scenario

Part A reuses the **exact two-moons dataset from L23** -- same `make_moons`, same seed, same noise --
so the rematch is fair: L23's K-Means scored ARI 0.239 on these crescents. ARI (the Adjusted Rand
Index) is again our yardstick: 1.0 is a perfect match to the true grouping, 0 is chance level.

Part B uses **real records**: the locations of every robbery and every vehicle theft reported to the
Puerto Rico Police from 2015 to 2018 (public incident records), about 26,600 points. The file we ship
keeps only four columns -- latitude, longitude, category and year -- with coordinates rounded to 3
decimals (about 100 m), so no address or time is included. There are no true labels here: the
question is not "did we recover the answer?" but "where are the dense areas, and how does the answer
depend on what we call dense?"

## Setup

The setup is **two cells** (the pattern every lab uses). The first only installs; the second
imports, rebuilds the L23 moons with the K-Means baseline, and loads the crime file (on Colab it is
downloaded from the course repository the first time).

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships scikit-learn, scipy, pandas, numpy and matplotlib, so this is a no-op there.
%pip install -q scikit-learn scipy
# local, in a terminal (not in the notebook):  uv add scikit-learn scipy pandas numpy matplotlib
```

```python
# Setup, cell 2 of 2 -- IMPORTS, DATA (safe to re-run without re-installing)
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import (DBSCAN, HDBSCAN, KMeans,
                             AgglomerativeClustering)
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Part A: the exact two crescents from L23 (same seed)
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
km = KMeans(n_clusters=2, random_state=42).fit(X)
km_ari = adjusted_rand_score(y, km.labels_)
print(f"L23 recap -- K-Means on the moons: ARI {km_ari:.3f}")

# Part B: robbery and vehicle-theft locations, Puerto Rico 2015-2018
CRIME_CSV = Path("data/l24_pr_robbery_vehicle_2015_2018.csv")
CRIME_URL = "https://raw.githubusercontent.com/devomh/cdat4011-2026/main/u06_unsupervised_learning/data/l24_pr_robbery_vehicle_2015_2018.csv"
if not CRIME_CSV.exists():   # Colab: download it once
    CRIME_CSV.parent.mkdir(exist_ok=True)
    urlretrieve(CRIME_URL, CRIME_CSV)
crimes = pd.read_csv(CRIME_CSV)
print("crime records:", len(crimes))
```

~~~text
L23 recap -- K-Means on the moons: ARI 0.239
crime records: 26646
~~~

## Part A -- Shapes K-Means Cannot Cluster

### Step 1: DBSCAN Wraps the Crescents

K-Means failed because it groups by **distance to a center**, and a crescent has no useful center.
**DBSCAN** changes the definition: a cluster is a **dense, connected region**. It walks from point
to point through dense neighborhoods, so it follows each crescent around its curve. It takes two
knobs -- `eps` (how close counts as "neighbor") and `min_samples` (how many points within `eps`
make a point **core**, the point itself included) -- and **no `k`**. Run it on the same moons:

```python
db = DBSCAN(eps=0.15, min_samples=5).fit(X)
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise = int((db.labels_ == -1).sum())
print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
print(f"DBSCAN ARI vs the true crescents: {adjusted_rand_score(y, db.labels_):.3f}")
print(f"(same data, K-Means scored {km_ari:.3f})")
```

~~~text
DBSCAN: 2 clusters, 0 noise points
DBSCAN ARI vs the true crescents: 1.000
(same data, K-Means scored 0.239)
~~~

DBSCAN found the two crescents **perfectly (ARI 1.000)** on the exact data where K-Means scored
0.239 -- and it was never told there were two clusters; it discovered that from the density. Plot
the result and the two arcs come out cleanly separated:

```python
plt.figure(figsize=(7, 3.5))
plt.scatter(X[:, 0], X[:, 1], c=db.labels_, s=8, cmap="tab10")
plt.title("DBSCAN: the two crescents, recovered")
plt.xlabel("x1"); plt.ylabel("x2"); plt.show()
```

### Step 2: Core, Border, Noise -- and the eps Dial

DBSCAN sorts every point into one of three roles. A **core** point has at least `min_samples`
points within `eps`. A **border** point is not core itself, but lies within `eps` of a core point,
so it joins that point's cluster. A **noise** point is neither and gets the label **-1**. In Step 1
every point was core -- the moons are evenly sampled -- so let's make the roles visible: inject 15
scattered outliers and use a slightly stricter density rule (`eps=0.10`, `min_samples=10`).

```python
rng = np.random.RandomState(42)
outliers = rng.uniform(low=[-1.5, -1.0], high=[2.5, 1.5], size=(15, 2))
X_out = np.vstack([X, outliers])
db2 = DBSCAN(eps=0.10, min_samples=10).fit(X_out)

is_core = np.zeros(len(X_out), dtype=bool)
is_core[db2.core_sample_indices_] = True
is_noise = db2.labels_ == -1
is_border = ~is_core & ~is_noise
print(f"clusters: {db2.labels_.max() + 1}")
print(f"core {is_core.sum()}, border {is_border.sum()}, noise {is_noise.sum()}")
print(f"injected outliers labelled noise: {is_noise[-15:].sum()} of 15")
print(f"ARI on the 1000 moon points: "
      f"{adjusted_rand_score(y, db2.labels_[:1000]):.3f}")
```

~~~text
clusters: 2
core 961, border 39, noise 15
injected outliers labelled noise: 12 of 15
ARI on the 1000 moon points: 0.994
~~~

All three roles now appear: 961 core points, 39 border points, 15 noise points. Twelve of the 15
injected outliers are flagged as noise; the other 3 happened to land close enough to a crescent to
join it. The remaining 3 noise points are genuine moon points on a sparse fringe, which is why the
ARI on the moons dips from 1.000 to 0.994. Plot the roles:

```python
plt.figure(figsize=(8, 3.5))
plt.scatter(*X_out[is_core].T, s=6, c="tab:blue", label="core")
plt.scatter(*X_out[is_border].T, s=18, c="tab:orange", label="border")
plt.scatter(*X_out[is_noise].T, s=40, c="black", marker="x", label="noise (-1)")
plt.title("DBSCAN roles (eps=0.10, min_samples=10)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5)); plt.show()
```

The orange border points sit on the thin edges of the crescents, where a point has some neighbors
but fewer than 10 within `eps`. The crosses are the points no dense region reaches -- DBSCAN's free
**anomaly detection**. K-Means has no such option: it would force every outlier into a cluster.

How sensitive is the result to `eps`? Sweep it on the original moons and watch the failure on each
side of the range that works:

```python
# Uncomment and complete: two values too small, two in the working range, one too large.
# for eps in (____, ____, ____, ____, ____):
#     lab = DBSCAN(eps=eps, min_samples=5).fit(X).labels_
#     nc = len(set(lab)) - (1 if -1 in lab else 0)
#     print(f"  eps={eps:.2f}: clusters {nc}, noise {int((lab == -1).sum())}, "
#           f"ARI {adjusted_rand_score(y, lab):.3f}")
```

<details><summary>Expected Output</summary>

~~~text
  eps=0.02: clusters 16, noise 908, ARI 0.002
  eps=0.05: clusters 7, noise 77, ARI 0.306
  eps=0.10: clusters 2, noise 1, ARI 0.998
  eps=0.30: clusters 2, noise 0, ARI 1.000
  eps=0.50: clusters 1, noise 0, ARI 0.000
~~~

One solution is `for eps in (0.02, 0.05, 0.10, 0.30, 0.50):`. Any values that show both failures
work. Too **small** and few neighborhoods are dense: at 0.05 the crescents break into 7 pieces and
77 points become noise; at 0.02 almost everything is noise (908 of 1000). Too **large** (0.50) and
the gap between the crescents is bridged, so everything is one cluster (ARI 0.000). In between lies
a wide **working range**: every value from about 0.08 to 0.30 finds the 2 crescents with an ARI of
at least 0.98. The job is to land inside that range, not to hit one magic number.
</details>

`eps` is a **distance**, so it only means something when every feature is measured on a comparable
scale. Record `x1` in units 20 times smaller (the same moons, multiplied by 20 along `x1`) and try
to find a working `eps`:

```python
X_wide = X * [20, 1]   # same moons, x1 now spans about 60 units, x2 about 1.5
for eps in (0.15, 0.5, 1.0, 2.0):
    lab = DBSCAN(eps=eps, min_samples=5).fit(X_wide).labels_
    print(f"  unscaled, eps={eps:.2f}: ARI {adjusted_rand_score(y, lab):.3f}")
X_std = StandardScaler().fit_transform(X_wide)
lab = DBSCAN(eps=0.15, min_samples=5).fit(X_std).labels_
print(f"  standardized, eps=0.15: ARI {adjusted_rand_score(y, lab):.3f}")
```

~~~text
  unscaled, eps=0.15: ARI 0.008
  unscaled, eps=0.50: ARI 0.138
  unscaled, eps=1.00: ARI 0.000
  unscaled, eps=2.00: ARI 0.000
  standardized, eps=0.15: ARI 0.996
~~~

No `eps` rescues the stretched data. A radius small enough to keep the crescents apart vertically
is tiny compared with the gaps between points along the stretched `x1`, so the crescents fall
apart; a radius large enough to bridge those gaps also bridges the two crescents. After
`StandardScaler` puts both features on the same scale, a plain `eps=0.15` scores 0.996 again.
Scale first, then choose `eps`.

### Step 3: Hierarchical Clustering and the Dendrogram

The other route to new shapes is **agglomerative hierarchical clustering**: start with every point
as its own cluster, then repeatedly **merge the two closest clusters** until one remains. The full
history of merges is a tree -- a **dendrogram** -- and you **cut** it at a height to choose how many
clusters you get. Build one on a tiny 12-point set so the tree is legible, then cut it with
`fcluster`:

```python
X_small, y_small = make_blobs(n_samples=12, centers=3, cluster_std=0.6, random_state=42)
Z = linkage(X_small, method="ward")
print("last 3 merge heights:", Z[-3:, 2].round(2).tolist())

cut = 10
labels_cut = fcluster(Z, t=cut, criterion="distance")
print(f"cut at height {cut}: {labels_cut.max()} clusters -> {labels_cut.tolist()}")
print("ARI vs truth:", round(adjusted_rand_score(y_small, labels_cut), 3))

agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_small)
print("same as AgglomerativeClustering(n_clusters=3)?",
      adjusted_rand_score(labels_cut, agg) == 1.0)
```

~~~text
last 3 merge heights: [1.98, 19.6, 34.19]
cut at height 10: 3 clusters -> [1, 1, 3, 1, 2, 1, 2, 3, 2, 3, 3, 2]
ARI vs truth: 1.0
same as AgglomerativeClustering(n_clusters=3)? True
~~~

The early merges all happen below height 2; then the heights leap to 19.6 and 34.19. Those two tall
merges join groups that are far apart, so a cut anywhere in the big gap -- we chose 10 -- leaves
**3 clusters**, which match the truth exactly (ARI 1.0). `fcluster` performs the cut;
`AgglomerativeClustering(n_clusters=3)` gives the same grouping when you already know how many
clusters you want. What does "height" measure? It is the linkage's **merge cost**: for single,
complete and average linkage it is a distance between the two clusters (nearest, farthest, or
average pair); for **ward** it is a cost computed from how much the merge increases the
within-cluster variance. Either way, taller means "these two groups were further apart". Draw the
tree with the cut:

```python
plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.axhline(cut, color="red", linestyle="--", label=f"cut at height {cut}")
plt.title("Dendrogram (ward linkage): the cut crosses 3 branches")
plt.xlabel("point index"); plt.ylabel("merge height (ward cost)")
plt.legend(); plt.show()
```

### Step 4: Linkage Matters -- the Bake-Off

"Merge the two closest clusters" hides a choice: closest **how**? That is the **linkage** rule.
**Single** linkage uses the nearest pair of points between two clusters, so it can chain along a
thin connected shape like a crescent. **Complete** uses the farthest pair, **average** the mean
over all pairs, and **ward** merges whichever pair least increases the within-cluster variance --
the last three all prefer compact groups. Put every contender on the moons:

```python
results = {
    "K-Means (k=2)": km.labels_,
    "DBSCAN (eps=0.15)": db.labels_,
}
for method in ("single", "complete", "average", "ward"):
    agg = AgglomerativeClustering(n_clusters=2, linkage=method)
    results[f"Agglomerative ({method})"] = agg.fit_predict(X)
for name, lab in results.items():
    print(f"  {name:26s} ARI {adjusted_rand_score(y, lab):.3f}")
ward_vs_km = adjusted_rand_score(km.labels_, results["Agglomerative (ward)"])
print(f"ward vs K-Means (do they agree with each other?): ARI {ward_vs_km:.3f}")
```

~~~text
  K-Means (k=2)              ARI 0.239
  DBSCAN (eps=0.15)          ARI 1.000
  Agglomerative (single)     ARI 1.000
  Agglomerative (complete)   ARI 0.432
  Agglomerative (average)    ARI 0.506
  Agglomerative (ward)       ARI 0.150
ward vs K-Means (do they agree with each other?): ARI 0.656
~~~

The split is about **shape, not effort**. DBSCAN (density) and single linkage (nearest-pair
chaining) both wrap the crescents perfectly. The three compact-group linkages all fail, by different
amounts: average 0.506, complete 0.432, and ward lowest at 0.150 -- below K-Means' 0.239. Ward and
K-Means share the same bias toward compact, round groups, but they do not give the same answer:
compared with each other their labels agree at ARI 0.656 only. Each draws its own border across the
crescents, and neither follows them. The lesson of Part A in one table: **match the algorithm to
the shape of your clusters.**

## Part B -- Where Do Robberies Concentrate?

### Step 5: The Data on a Map

Each row is one reported incident: where it happened (latitude, longitude), what it was, and the
year. Count them, then plot every robbery. We draw longitude across and latitude up, and stretch the
axes so that 1 km is the same length both ways (one degree of longitude is shorter than one degree
of latitude at Puerto Rico's 18 degrees north).

```python
print(crimes.head(3))
print(crimes.groupby(["category", "year"]).size().unstack())
robbery = crimes[crimes["category"] == "robbery"].reset_index(drop=True)
print("robberies:", len(robbery))
```

~~~text
      lat     lon category  year
0  17.937 -66.194  robbery  2015
1  17.938 -66.145  robbery  2015
2  17.949 -66.127  robbery  2015
year           2015  2016  2017  2018
category                             
robbery        4051  3036  2865  2226
vehicle theft  4018  3580  3275  3595
robberies: 12178
~~~

```python
LAT0 = np.radians(18.2)   # Puerto Rico's latitude


def crime_map(ax, df, labels=None, title=""):
    """Grey dots = every record; colored dots = points inside a cluster."""
    lon, lat = df["lon"].to_numpy(), df["lat"].to_numpy()
    ax.scatter(lon, lat, s=1, c="0.8")
    if labels is not None:
        inside = labels >= 0
        ax.scatter(lon[inside], lat[inside], s=2, c=labels[inside] % 20,
                   cmap="tab20", vmin=0, vmax=19)
    ax.set_aspect(1 / np.cos(LAT0))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


fig, ax = plt.subplots(figsize=(10, 4))
crime_map(ax, robbery, title="Robberies, Puerto Rico 2015-2018 (one dot each)")
plt.show()
```

With no basemap at all, the dots draw the island: the dense northeast is the San Juan metropolitan
area, and the towns show up as smaller knots along the coast and in the interior. We pool the four
years into one picture. Do not compare years with this file: the share of records the police could
place on a map changed from year to year, so a drop in one area may be a mapping gap rather than
fewer robberies.

### Step 6: Choosing eps -- and Watching Hotspots Appear

Latitude and longitude are angles, not lengths, so Euclidean distance on them is wrong. This is the
`metric=` argument L23 promised: DBSCAN accepts `metric="haversine"`, the great-circle distance on a
sphere. It expects `(lat, lon)` in **radians** and returns distances in radians, so we write `eps`
in metres and divide by the Earth's radius.

How large should `eps` be? There is no answer key to sweep against here, so we use the standard
diagnostic, the **k-distance plot**. For every point, measure the distance to its 25th nearest
neighbor (counting the point itself, exactly as `min_samples=25` does). A point is core at a given
`eps` precisely when that distance is at most `eps`. Sort the distances and plot them:

```python
EARTH_R = 6_371_000   # metres
rob_rad = np.radians(robbery[["lat", "lon"]].to_numpy())

nn = NearestNeighbors(n_neighbors=25, metric="haversine").fit(rob_rad)
dist, _ = nn.kneighbors(rob_rad)
k_dist = dist[:, -1] * EARTH_R   # distance to the 25th neighbor, in metres
for eps_m in (500, 1000, 2000):
    print(f"eps = {eps_m:4d} m: {np.mean(k_dist <= eps_m):.0%} of robberies are core")

plt.figure(figsize=(7, 3.5))
plt.plot(np.sort(k_dist))
for eps_m in (500, 1000, 2000):
    plt.axhline(eps_m, color="0.5", linestyle="--", linewidth=1)
plt.yscale("log")
plt.xlabel("robberies, sorted"); plt.ylabel("distance to 25th neighbor (m)")
plt.title("k-distance plot (k = 25)"); plt.show()
```

~~~text
eps =  500 m: 39% of robberies are core
eps = 1000 m: 65% of robberies are core
eps = 2000 m: 81% of robberies are core
~~~

Textbook examples show a sharp "knee" in this curve, and you pick `eps` there. Real data rarely
gives one. Here the curve climbs smoothly from about 100 m to a few kilometres and only shoots up at
the far right, where isolated rural robberies have no 25 neighbors within 10 km. So the plot does
not give one answer, but it does give a scale: `eps` values from about 500 m to 2 km cover the
middle of the curve, and the printout says how much of the data each one treats as dense. On data
with no answer key, `eps` is a choice of **what you want to call a hotspot**. The next cell makes
that choice visible.

Now run DBSCAN at three values of `eps` (with `min_samples=25`) and map each result:

```python
def summarize(labels):
    sizes = np.bincount(labels[labels >= 0])
    return (f"{len(sizes)} clusters, {np.mean(labels == -1):.0%} noise, "
            f"largest holds {sizes.max() / len(labels):.0%}")


fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for ax, eps_m in zip(axes, (500, 1000, 2000)):
    labels = DBSCAN(eps=eps_m / EARTH_R, min_samples=25,
                    metric="haversine").fit_predict(rob_rad)
    print(f"eps = {eps_m:4d} m: {summarize(labels)}")
    crime_map(ax, robbery, labels, f"DBSCAN, eps = {eps_m} m, min_samples = 25")
plt.tight_layout(); plt.show()
```

~~~text
eps =  500 m: 39 clusters, 50% noise, largest holds 11%
eps = 1000 m: 36 clusters, 27% noise, largest holds 49%
eps = 2000 m: 27 clusters, 13% noise, largest holds 66%
~~~

Read the three maps from top to bottom. At **500 m** only the densest knots qualify: 39 tight
hotspots, most of them inside metropolitan San Juan, plus the centers of Caguas, Ponce, Mayaguez
and a few other towns; half of all robberies are noise. At **1 km** the town hotspots grow and new ones appear
along the coasts, but the metro hotspots have fused into one cluster holding 49% of all robberies.
At **2 km** almost every town has a cluster, yet the metro cluster has swallowed 66% of the data and
stretches south toward Caguas -- one "hotspot" covering the most populated part of the island is no
longer a useful answer.

### Step 7: HDBSCAN -- One Answer for a City and a Town

DBSCAN's weak spot is now visible: **one global `eps` cannot fit two densities**. A radius small
enough to separate hotspots inside metropolitan San Juan is too small for any small town to have a
dense core; a radius large enough for the towns fuses the whole metro area. **HDBSCAN**
(Hierarchical DBSCAN, in scikit-learn since version 1.3) removes `eps`: in effect it runs DBSCAN
at every radius at once, builds the tree of how clusters split as the radius shrinks, and keeps
the clusters that persist longest. Its main knob is `min_cluster_size`, the smallest group you
would call a hotspot.

```python
# copy=True: work on a copy of rob_rad (and silence a version warning)
hdb = HDBSCAN(min_cluster_size=100, metric="haversine", copy=True)
hdb_labels = hdb.fit_predict(rob_rad)
print("HDBSCAN:", summarize(hdb_labels))

fig, ax = plt.subplots(figsize=(10, 4))
crime_map(ax, robbery, hdb_labels, "HDBSCAN, min_cluster_size = 100")
plt.show()
```

~~~text
HDBSCAN: 16 clusters, 50% noise, largest holds 9%
~~~

HDBSCAN gives both answers at once. The metro area is split into several hotspots (around San
Juan, Bayamon, Carolina and Toa Baja), and the largest holds only 9% of the robberies. At the same
time it keeps clusters in towns that are much sparser -- Caguas, Ponce, Mayaguez, Aguadilla,
Arecibo, Humacao, Cayey, Guayama -- each at its own density. In sparse areas a "hotspot" found this
way is a broad region, not a few blocks: the Ponce cluster runs along much of the south coast. The
definition still decides what counts. Half of the robberies are still noise:
they happened somewhere, but not in any concentration of at least 100 nearby incidents. Raise
`min_cluster_size` and only the biggest hotspots survive; lower it and smaller ones appear.

### Step 8: Hierarchical Clustering on Hotspots -- the Wrong Tool

Could agglomerative clustering find these hotspots? Try both ends of the linkage scale. First,
**single** linkage cut at 1 km, which `AgglomerativeClustering` can do directly with
`distance_threshold` and the same haversine metric. Compare it with DBSCAN at `eps` = 1 km and
`min_samples=1` -- a rule under which every point is core:

```python
single = AgglomerativeClustering(
    n_clusters=None, distance_threshold=1000 / EARTH_R,
    linkage="single", metric="haversine").fit_predict(rob_rad)
db_one = DBSCAN(eps=1000 / EARTH_R, min_samples=1,
                metric="haversine").fit_predict(rob_rad)
print("single-linkage groups:", single.max() + 1,
      "| DBSCAN(min_samples=1) clusters:", db_one.max() + 1)
print("same grouping? ARI", round(adjusted_rand_score(single, db_one), 3))
sizes = np.bincount(single)
print(f"largest group: {sizes.max() / len(single):.0%} of robberies; "
      f"groups with a single robbery: {np.sum(sizes == 1)}")
```

~~~text
single-linkage groups: 693 | DBSCAN(min_samples=1) clusters: 693
same grouping? ARI 1.0
largest group: 56% of robberies; groups with a single robbery: 371
~~~

The two groupings are **identical** (ARI 1.0). That is not a coincidence: cutting a single-linkage
tree at height `eps` joins any two points connected by a chain of steps shorter than `eps`, and
that is exactly DBSCAN when `min_samples=1` makes every point core. Seen this way, `min_samples` is
what DBSCAN adds to single linkage -- a **density test** that stops a chain from running through
sparse points. Without it the chain runs on: one group spans the metro area from Dorado to
Rio Grande, about 50 km, and holds 56% of all robberies, while 371 robberies sit alone in groups of
one. This is single linkage's **chaining problem** on real data.

Second, **ward** linkage. Ward is defined by variance, so it only works with plain Euclidean
coordinates; we flatten latitude and longitude into metres (accurate enough for an island 180 km
wide) and ask for 12 clusters:

```python
xy = np.column_stack([rob_rad[:, 1] * EARTH_R * np.cos(LAT0),
                      rob_rad[:, 0] * EARTH_R])
ward = AgglomerativeClustering(n_clusters=12, linkage="ward").fit_predict(xy)
print("ward cluster sizes:", np.bincount(ward).tolist())
print("noise points:", int(np.sum(ward == -1)))

big = np.isin(single, np.flatnonzero(sizes >= 25))   # show groups of 25+
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
crime_map(axes[0], robbery, np.where(big, single, -1),
          "single linkage cut at 1 km (groups of 25+ shown)")
crime_map(axes[1], robbery, ward, "ward linkage, 12 clusters")
plt.tight_layout(); plt.show()
```

~~~text
ward cluster sizes: [714, 788, 2195, 1236, 514, 349, 4012, 305, 201, 1179, 310, 375]
noise points: 0
~~~

Ward answers a different question. It cuts the whole island into 12 compact regions and puts
**every** robbery in one of them -- 0 noise points -- including the isolated ones in the countryside.
That is a reasonable way to draw **districts** (for example, to split patrol areas), but it is not
hotspot detection: a hotspot is a place that is *denser than its surroundings*, and ward has no idea
of "not dense enough". Hierarchical clustering also gets expensive quickly: it needs the distance
between every pair of points, so the cost grows like $n^2$. With 12,178 points it is fine; with a
year of every crime in Puerto Rico (about 80,000 records) the pairwise distances alone would need
tens of gigabytes. For hotspots, density methods are the right tool.

## Your Turn

### Exercise 1 -- DBSCAN on nested rings

Here are two concentric rings (`make_circles`) -- an even harder shape than the moons. K-Means is
printed for you; it is helpless. Cluster the rings with DBSCAN and recover both.

```python
X_ring, y_ring = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
km_ring = KMeans(n_clusters=2, random_state=42).fit(X_ring)
print("rings:", X_ring.shape, "-- K-Means ARI:",
      round(adjusted_rand_score(y_ring, km_ring.labels_), 3))
```

~~~text
rings: (1000, 2) -- K-Means ARI: -0.001
~~~

**Hint:** DBSCAN takes no `k` -- only `eps` and `min_samples`. The rings were drawn with the same
noise level as the moons, so the moons' working range from Step 2 is a good place to start. Count
clusters from `labels_` (leave out -1) and score with `adjusted_rand_score(y_ring, ...)`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
  eps=0.10: 2 clusters, ARI 1.000
  eps=0.15: 2 clusters, ARI 1.000
  eps=0.20: 2 clusters, ARI 1.000
~~~

One solution tries three values from the moons' working range, with `min_samples=5`:

~~~python
for eps in (0.10, 0.15, 0.20):
    lab = DBSCAN(eps=eps, min_samples=5).fit(X_ring).labels_
    nc = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"  eps={eps:.2f}: {nc} clusters, "
          f"ARI {adjusted_rand_score(y_ring, lab):.3f}")
~~~

Any of them recovers both rings perfectly. K-Means scores ARI -0.001 -- chance level, because
the two rings share the same center, so no centroid split can separate them. DBSCAN ignores
centers entirely and walks each ring's dense band. Concentric structure is invisible to a centroid
method and obvious to a density method.
</details>

### Exercise 2 -- Linkage face-off on the rings

Now cluster the same rings with agglomerative hierarchical clustering, **single** versus **ward**
linkage, and report both ARIs. Which linkage follows a ring, and why?

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
  linkage=single  ARI 1.000
  linkage=ward    ARI 0.004
~~~

Single linkage scores **1.000**, ward **0.004**. Single linkage merges on the nearest pair of
points, so it chains all the way around each ring's connected band -- exactly the trick that wrapped
the moons. Ward insists on compact, low-variance groups, so it carves the rings into two halves of a
pie and fails, as it did on the crescents. Connected structure needs a rule that follows
connections.
</details>

### Exercise 3 -- Do vehicle thefts concentrate in the same places?

Repeat Steps 6-7 on the **vehicle thefts** in `crimes`. First print the k-distance core shares at
500 m, 1 km and 2 km (`n_neighbors=25`) and compare them with the robberies'. Then run DBSCAN with
`eps` = 1 km and `min_samples=25`, and HDBSCAN with `min_cluster_size=100`; print `summarize(...)`
for each, and map the HDBSCAN result next to the robbery one. Are vehicle thefts more or less
concentrated than robberies, and do the same places light up?

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
eps =  500 m: 55% of vehicle thefts are core
eps = 1000 m: 74% of vehicle thefts are core
eps = 2000 m: 86% of vehicle thefts are core
vehicle theft, DBSCAN 1 km: 31 clusters, 20% noise, largest holds 56%
vehicle theft, HDBSCAN:    18 clusters, 39% noise, largest holds 24%
~~~

One solution:

~~~python
vehicle = crimes[crimes["category"] == "vehicle theft"].reset_index(drop=True)
veh_rad = np.radians(vehicle[["lat", "lon"]].to_numpy())
nn_veh = NearestNeighbors(n_neighbors=25, metric="haversine").fit(veh_rad)
k_veh = nn_veh.kneighbors(veh_rad)[0][:, -1] * EARTH_R
for eps_m in (500, 1000, 2000):
    print(f"eps = {eps_m:4d} m: {np.mean(k_veh <= eps_m):.0%} of vehicle thefts are core")
db_veh = DBSCAN(eps=1000 / EARTH_R, min_samples=25,
                metric="haversine").fit_predict(veh_rad)
print("vehicle theft, DBSCAN 1 km:", summarize(db_veh))
hdb_veh = HDBSCAN(min_cluster_size=100, metric="haversine",
                  copy=True).fit_predict(veh_rad)
print("vehicle theft, HDBSCAN:   ", summarize(hdb_veh))
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
crime_map(axes[0], robbery, hdb_labels, "robbery, HDBSCAN")
crime_map(axes[1], vehicle, hdb_veh, "vehicle theft, HDBSCAN")
plt.tight_layout(); plt.show()
~~~

Vehicle thefts are **more concentrated** than robberies: at every radius a larger share of them is
core (55% / 74% / 86% at 500 m / 1 km / 2 km, vs 39% / 65% / 81%), fewer of them are noise (20% vs 27% at 1 km; 39% vs 50%
with HDBSCAN), and the core of San Juan stays one large cluster (24% of all vehicle thefts) where
the robbery map splits it into three. Bayamon, Carolina and Levittown are separate hotspots for both
crimes. And largely the same places light up: the metro area, Caguas, Ponce, Cayey, Arecibo and the
west coast. Vehicle thefts add hotspots along the north-coast towns west of San Juan and form one long strip on the west coast from Aguadilla to
Mayaguez, while Humacao and Guayama, robbery hotspots, do not show up for vehicles. Clustering
cannot say *why* -- parking, traffic, policing and reporting all differ -- but it tells you where
to look.
</details>

## Summary

- K-Means drew one straight border across L23's crescents (ARI 0.239); **DBSCAN** clusters by
  **connected density** and wrapped them perfectly (ARI 1.000) with no `k`.
- DBSCAN sorts points into **core / border / noise (label -1)**; noise is free **anomaly
  detection** (12 of 15 injected outliers flagged). Its dial `eps` has a **working range** (about
  0.08-0.30 on the moons) with a failure on each side: too small fragments the data into noise, too
  large fuses everything. Set it on **scaled** features -- after stretching one feature, no `eps`
  worked until `StandardScaler`.
- **Agglomerative** clustering merges the two closest clusters into a **dendrogram**; `fcluster`
  **cuts** it at a height (the linkage's merge cost) to give clusters -- 3 on the 12-point set.
- **Linkage** decides the shape: **single** chains along crescents and rings (ARI 1.000); complete,
  average and **ward** prefer compact groups and fail (0.432 / 0.506 / 0.150). Ward shares K-Means'
  bias but not its answer (they agree at ARI 0.656).
- On real robbery locations, **haversine** DBSCAN with a **k-distance plot** finds hotspots whose
  number and size depend on `eps`; one global `eps` cannot fit a dense metro area and sparse towns
  at once. **HDBSCAN** can: 16 hotspots, the largest only 9% of the data.
- Single linkage cut at `eps` **is** DBSCAN with `min_samples=1` -- and it chains (one group = 56% of
  robberies); ward tiles the island into districts with no noise. Hierarchical clustering is the
  wrong tool for hotspots, and its pairwise-distance cost grows like $n^2$.
- Next, L25 closes Unit VI's clustering trio with Gaussian mixtures and EM -- soft, probabilistic
  clustering that gives each point a membership probability rather than a single hard label.
