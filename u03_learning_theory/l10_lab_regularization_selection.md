---
title: "Lab: Which Sensors Actually Matter"
unit: "III"
lesson: "10"
type: lab
tags: [regularization, ridge, lasso, feature-selection, scikit-learn]
difficulty: intermediate
duration: "45 mins"
---

**Goal:** watch regularization turn an overfit model into a trustworthy one -- and watch
Lasso decide, on its own, which features to keep. You will fit ordinary least squares,
add the Ridge and Lasso penalties, choose the penalty strength by cross-validation
(the L09 protocol), and read off the handful of sensors Lasso kept. Pairs with the
concept note
[Regularization and Model/Feature Selection](l10_concept_regularization_selection.qmd).

> **Previously:** L09 -- The Bias-Variance Dilemma & Cross-Validation  |  **Next:** L11 -- Evaluation Metrics

> This page is the read-only view. To run the lab, open the notebook (`l10_lab_regularization_selection.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u03_learning_theory/l10_lab_regularization_selection.ipynb)

## Scenario

A field station rings a coqui habitat with **30 environmental sensors** -- temperature,
humidity, soil moisture, light, wind, and two dozen more -- and logs each night's
**chorus loudness** in decibels. The catch: only a *few* of the 30 sensors actually
drive the chorus; the rest are noise. You do not know which. Build a model that predicts
loudness, and let regularization tell you which sensors earn their place.

The data is **synthetic** (a fixed seed, identical for everyone) with *fictionalized but
plausible* values -- the framing is real, the numbers are not field measurements.

## Setup

The setup is **two cells** (the pattern every lab uses). The first only installs; the
second imports, seeds the generator, builds the data, and splits it.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Colab already ships scikit-learn, numpy, pandas and matplotlib, so this is effectively a no-op there.
%pip install -q scikit-learn
# local, in a terminal (not in the notebook):  uv add scikit-learn numpy pandas matplotlib
```

```python
# Setup, cell 2 of 2 -- IMPORTS, SEED, DATA (safe to re-run without re-installing)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 30 sensors, only 5 of them truly drive the chorus loudness (we will not peek at which).
X, y, true_coef = make_regression(n_samples=90, n_features=30, n_informative=5,
                                  noise=40.0, random_state=7, coef=True)
feat = [f"s{i:02d}" for i in range(30)]          # sensor names s00 .. s29
sensors = pd.DataFrame(X.round(3), columns=feat)  # already on one scale (no rescaling needed here)
loudness = y.round(2)                             # nightly chorus loudness (dB)

X_train, X_test, y_train, y_test = train_test_split(sensors, loudness, test_size=0.30,
                                                    random_state=7)
print(len(X_train), "train nights /", len(X_test), "test nights /", sensors.shape[1], "sensors")
```

~~~text
63 train nights / 27 test nights / 30 sensors
~~~

(The features `make_regression` produces are already standardized, so this lab skips
the `StandardScaler` step the concept note insists on for real data -- a deliberate
shortcut so the penalties are the only moving part.)

## Step 1: Ordinary Least Squares Overfits

Fit plain linear regression -- no penalty -- and score it on both halves:

```python
from sklearn.linear_model import LinearRegression

ols = LinearRegression().fit(X_train, y_train)
print(f"OLS train R2: {ols.score(X_train, y_train):.3f}")
print(f"OLS test  R2: {ols.score(X_test, y_test):.3f}")
print(f"sensors used (nonzero weights): {int((np.abs(ols.coef_) > 1e-6).sum())} of 30")
```

~~~text
OLS train R2: 0.918
OLS test  R2: 0.658
sensors used (nonzero weights): 30 of 30
~~~

A textbook overfit: 0.918 on the data it trained on, but only 0.658 on nights it has
never seen -- a 0.26 generalization gap (exactly the diagnosis L09 taught). And it leans
on **all 30 sensors**, including the two dozen that are pure noise. The model memorized
the training nights through features that mean nothing.

## Step 2: Add a Penalty

### Ridge (L2): shrink every weight

Ridge taxes the sum of squared weights. It cannot zero a feature, but it pulls every
coefficient toward zero -- a smaller, calmer model:

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
print(f"Ridge test R2: {ridge.score(X_test, y_test):.3f}")
print(f"sensors used:  {int((np.abs(ridge.coef_) > 1e-6).sum())} of 30")

norm_1  = np.linalg.norm(Ridge(alpha=1).fit(X_train, y_train).coef_)
norm_50 = np.linalg.norm(Ridge(alpha=50).fit(X_train, y_train).coef_)
print(f"coefficient size (L2 norm): alpha=1 -> {norm_1:.1f},  alpha=50 -> {norm_50:.1f}")
```

~~~text
Ridge test R2: 0.674
sensors used:  30 of 30
coefficient size (L2 norm): alpha=1 -> 94.0,  alpha=50 -> 53.4
~~~

Ridge nudges the test score up (0.658 to 0.674) and visibly **shrinks** the weights
(their combined length falls from 94 to 53 as the dial turns) -- but all 30 sensors are
still in the model. Ridge tames; it does not select.

### Lasso (L1): zero the weak ones -- completion problem

Lasso taxes the sum of *absolute* weights, which drives weak coefficients to **exactly
zero**. Complete the marked lines and watch the sensor count collapse:

```python
from sklearn.linear_model import Lasso

# Uncomment and complete the marked lines:
# lasso = Lasso(alpha=____, max_iter=100000).fit(X_train, y_train)   # try alpha=5
# print(f"Lasso test R2: {lasso.score(____, ____):.3f}")            # score on the TEST half
# print(f"sensors used:  {int((np.abs(lasso.coef_) > 1e-6).sum())} of 30")  # how many survived?
```

<details><summary>Expected Output</summary>

~~~text
Lasso test R2: 0.806
sensors used:  9 of 30
~~~

Lasso does what Ridge could not: the test score jumps to **0.806** (from OLS's 0.658)
*and* 21 of the 30 sensors are switched off entirely. One penalty bought both a better
forecast and a far simpler, more interpretable model -- nine sensors instead of thirty.
</details>

## Step 3: Choose the Penalty Strength by Cross-Validation

`alpha` is a hyperparameter, so you choose it the L09 way -- by cross-validation on the
training data, never on the test set:

```python
from sklearn.model_selection import cross_val_score

for a in (0.1, 1, 5, 10, 20):
    cv = cross_val_score(Lasso(alpha=a, max_iter=100000), X_train, y_train,
                         cv=5, scoring="r2").mean()
    print(f"alpha={a}: CV R2 {cv:.3f}")
```

~~~text
alpha=0.1: CV R2 0.524
alpha=1: CV R2 0.638
alpha=5: CV R2 0.786
alpha=10: CV R2 0.778
alpha=20: CV R2 0.709
~~~

The CV curve is the bias-variance U turned on its side: too little penalty (alpha=0.1)
leaves the overfit in place, too much (alpha=20) crushes useful weights and underfits,
and the sweet spot sits at **alpha=5** (CV R2 0.786). That is the value you would refit
on all the training data and carry to the single test score.

## Your Turn

### Exercise 1 -- The regularization path

Walk `alpha` across `(1, 5, 10)` for Lasso. For each, print how many sensors survive and
the test R2. What is the trade as the penalty grows?

**Hint:** loop over the three alphas; for each, fit `Lasso(alpha=a, max_iter=100000)` on the training data and print `int((np.abs(lasso.coef_) > 1e-6).sum())` and `lasso.score(X_test, y_test)`.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
alpha=1: 22 sensors, test R2 0.747
alpha=5: 9 sensors, test R2 0.806
alpha=10: 5 sensors, test R2 0.806
~~~

Turning the dial up sweeps the model from 22 sensors down to 5. The test score holds
near its best from alpha=5 to alpha=10 even as four more sensors drop -- so the simpler
5-sensor model is essentially free. That trajectory, sparse to sparser, is the
**regularization path**.
</details>

### Exercise 2 -- Did Lasso find the real sensors?

The data was built so only 5 of the 30 sensors truly matter; their names are hiding in
`true_coef`. Reveal them, then compare to the sensors Lasso kept at `alpha=5`.

**Hint:** the true sensors are the `feat` names at `np.flatnonzero(true_coef)`; refit `Lasso(alpha=5, max_iter=100000)` and read its kept sensors the same way (the `feat` names where `|coef_| > 1e-6`), then compare the two sets.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
truly informative sensors: ['s02', 's16', 's17', 's25', 's26']
Lasso(alpha=5) kept: ['s05', 's06', 's13', 's16', 's17', 's25', 's26', 's28', 's29']
recovered 4 of 5 true sensors; dropped 20 of 25 noise sensors
~~~

Not perfect -- Lasso missed the weakest true sensor (s02) and kept five noise sensors --
but out of 30 candidates it recovered most of the real signal and threw out 20 of the 25
distractors, with no idea in advance which were which. That is feature selection done by
the model, for free, as a side effect of the penalty.
</details>

## Summary

- Ordinary least squares overfit (train 0.918, test 0.658) and leaned on all 30 sensors,
  noise included -- the disease L09 taught you to diagnose.
- Ridge (L2) shrank every weight (L2 norm 94 to 53) and nudged the test score to 0.674,
  but kept all 30 features: it tames, it does not select.
- Lasso (L1) drove weak weights to exactly zero -- test R2 0.806 with only 9 sensors --
  curing the overfit and selecting features in one move.
- The penalty strength `alpha` is a hyperparameter chosen by cross-validation (CV picked
  alpha=5); the regularization path trades features for almost no accuracy.
- Lasso recovered 4 of the 5 true sensors and dropped 20 of 25 noise sensors -- embedded
  feature selection, measured.
- Next (L11): you have been scoring everything with R2 and accuracy -- but those hide how
  a model fails. Unit III closes with the metrics that show the full picture.
