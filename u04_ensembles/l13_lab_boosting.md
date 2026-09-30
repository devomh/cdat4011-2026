---
title: "Lab: Chaining Weak Stumps"
unit: "IV"
lesson: "13"
type: lab
tags: [boosting, adaboost, gradient-boosting, learning-rate, scikit-learn]
difficulty: intermediate
duration: "55 mins"
---

**Goal:** first run both boosting algorithms by hand on the concept note's tiny toys, and
check that scikit-learn gets exactly your numbers. Then start from a decision stump that
barely beats a coin flip, and watch boosting chain hundreds of them into a classifier that
rivals the random forest. You will run
AdaBoost, tune a gradient-boosting model with the learning-rate dial, let early stopping
pick the number of trees, and finish by putting bagging and boosting head to head on the
*same* data. Pairs with the concept note [Boosting](l13_concept_boosting.qmd).

> **Previously:** L12 -- Decision Trees & Bagging  |  **Next:** L14 -- From Perceptrons to Neural Networks (Unit V)

> This page is the read-only view. To run the lab, open the notebook (`l13_lab_boosting.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4011-2026/blob/main/u04_ensembles/l13_lab_boosting.ipynb)

## Scenario

We reuse the **exact coqui acoustic dataset from L12** -- 600 calls, eight features, two
species, the same seed and the same train/test split. Using identical data is the point:
at the end we compare a random forest (bagging, from L12) against gradient boosting
(this lesson) on the same yardstick, so the difference is the *method*, not the data.

The data is **synthetic** (a fixed seed, identical for everyone) with *fictionalized but
plausible* values -- the framing is real, the numbers are not field measurements.

## Setup

The setup is **two cells** (the pattern every lab uses). The first only installs; the
second imports, seeds the generator, builds the data, and splits it.

```python
# Setup, cell 1 of 2 -- INSTALL (run once; Colab wipes installs when it resets on open)
# Pin the tested estimator API; scikit-learn installs its required numerical dependencies.
%pip install -q scikit-learn==1.9.0
# local, in a terminal (not in the notebook):  uv add scikit-learn==1.9.0 numpy pandas matplotlib
```

```python
# Setup, cell 2 of 2 -- IMPORTS, SEED, DATA (safe to re-run without re-installing)
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Identical to L12: same call, same seed, same split -> a fair bagging-vs-boosting test.
X, y = make_classification(n_samples=600, n_features=8, n_informative=4, n_redundant=1,
                           n_classes=2, class_sep=0.9, flip_y=0.05,
                           shuffle=False, random_state=11)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=11, stratify=y)
print(len(X_train), "train /", len(X_test), "test")
```

~~~text
420 train / 180 test
~~~

## Step 1: Boosting by Hand

Before the coqui data, run both algorithms on the two toys from the concept note, one
round at a time. First AdaBoost on the ten labeled points. The loop is the concept note's
Algorithm 1 line for line; the only new tool is `sample_weight`, which tells a
scikit-learn stump how much each point counts:

```python
# The 10-point toy from the concept note (labels +1 / -1)
X_toy = np.array([[1, 6], [2, 8], [3, 2], [4, 1],
                  [5, 3], [6, 10], [7, 7], [8, 5],
                  [9, 9], [10, 4]])
y_toy = np.array([1, -1, 1, -1, -1, -1, 1, 1, -1, 1])

w = np.full(10, 0.1)     # every point starts equal
score = np.zeros(10)     # the running weighted vote
for m in range(1, 4):
    st = DecisionTreeClassifier(max_depth=1,
                                random_state=0)
    st.fit(X_toy, y_toy, sample_weight=w)
    pred = st.predict(X_toy)
    miss = pred != y_toy
    r = w[miss].sum()              # weighted error
    alpha = np.log((1 - r) / r)    # the stump's vote
    w[miss] *= np.exp(alpha)       # upweight mistakes
    w /= w.sum()                   # renormalize
    score += alpha * pred
    # the stump's one question, e.g. "x2 <= 7.5"
    f = st.tree_.feature[0] + 1
    q = f"x{f} <= {st.tree_.threshold[0]}"
    missed = (np.where(miss)[0] + 1).tolist()
    errors = int((np.sign(score) != y_toy).sum())
    print(f"round {m}: {q}  missed {missed}")
    print(f"  r={r:.4f}  alpha={alpha:.3f}"
          f"  missed now hold {w[miss].sum():.2f}")
    print("  points 1-5: ",
          " ".join(f"{v:.4f}" for v in w[:5]))
    print("  points 6-10:",
          " ".join(f"{v:.4f}" for v in w[5:]))
    print(f"  ensemble errors after {m}: {errors}")
```

~~~text
round 1: x2 <= 7.5  missed [4, 5]
  r=0.2000  alpha=1.386  missed now hold 0.50
  points 1-5:  0.0625 0.0625 0.0625 0.2500 0.2500
  points 6-10: 0.0625 0.0625 0.0625 0.0625 0.0625
  ensemble errors after 1: 2
round 2: x1 <= 6.5  missed [1, 3, 9]
  r=0.1875  alpha=1.466  missed now hold 0.50
  points 1-5:  0.1667 0.0385 0.1667 0.1538 0.1538
  points 6-10: 0.0385 0.0385 0.0385 0.1667 0.0385
  ensemble errors after 2: 3
round 3: x1 <= 3.5  missed [2, 7, 8, 10]
  r=0.1538  alpha=1.705  missed now hold 0.50
  points 1-5:  0.0985 0.1250 0.0985 0.0909 0.0909
  points 6-10: 0.0227 0.1250 0.1250 0.0985 0.1250
  ensemble errors after 3: 0
~~~

Every number matches the concept note. Round 1 misses points 4 and 5, whose weight jumps
from 0.1 to 0.25 while the others drop to 0.0625. After *every* round the points just
missed hold exactly half the weight, so each stump is a coin flip on the next round's
weights, and the next stump has to ask a different question. Watch the last line too: two
rounds make **more** mistakes than one (the louder second vote wins every disagreement),
and the third stump brings the ensemble to zero errors.

Is scikit-learn's `AdaBoostClassifier` really this loop? Fit it with three stumps and
compare its vote weights with yours:

```python
from sklearn.ensemble import AdaBoostClassifier

ada_toy = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1, random_state=0),
    n_estimators=3, learning_rate=1.0, random_state=0,
).fit(X_toy, y_toy)
alphas = ada_toy.estimator_weights_.round(3).tolist()
print("sklearn alphas:", alphas)
print("sklearn train accuracy:", ada_toy.score(X_toy, y_toy))
```

~~~text
sklearn alphas: [1.386, 1.466, 1.705]
sklearn train accuracy: 1.0
~~~

The same three votes and a perfect score on the toy: the library is the loop you just
wrote. Now gradient boosting, on the six-point regression toy. Each round fits a
regression stump to the **residuals** and takes half a step (`nu = 0.5`):

```python
from sklearn.tree import DecisionTreeRegressor

# The 6-point regression toy from the concept note
x_gb = np.arange(1, 7).reshape(-1, 1)
y_gb = np.array([1.0, 2, 3, 10, 12, 14])
nu = 0.5                        # the learning rate

def sse(F):
    return float(((y_gb - F) ** 2).sum())

F = np.full(6, y_gb.mean())     # round 0: the mean
print(f"round 0: F={F.tolist()}")
print(f"  SSE={sse(F):.3f}")
for m in range(1, 4):
    res = y_gb - F              # what is still wrong
    h = DecisionTreeRegressor(max_depth=1,
                              random_state=0)
    h.fit(x_gb, res)            # fit the residuals
    F = F + nu * h.predict(x_gb)    # a half step
    print(f"round {m}: residuals {res.tolist()}")
    print(f"  split x <= {h.tree_.threshold[0]}"
          f"  SSE={sse(F):.3f}")
    print(f"  F={F.round(2).tolist()}")
```

~~~text
round 0: F=[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
  SSE=160.000
round 1: residuals [-6.0, -5.0, -4.0, 3.0, 5.0, 7.0]
  split x <= 3.5  SSE=47.500
  F=[4.5, 4.5, 4.5, 9.5, 9.5, 9.5]
round 2: residuals [-3.5, -2.5, -1.5, 0.5, 2.5, 4.5]
  split x <= 3.5  SSE=19.375
  F=[3.25, 3.25, 3.25, 10.75, 10.75, 10.75]
round 3: residuals [-2.25, -1.25, -0.25, -0.75, 1.25, 3.25]
  split x <= 4.5  SSE=7.984
  F=[2.69, 2.69, 2.69, 10.19, 11.88, 11.88]
~~~

Rounds 1 and 2 split at the same place, `x <= 3.5`: the first stump only closed half of
the gap between the two groups, so the second stump finishes that job. Only round 3 finds
new structure (`x <= 4.5`, the trend inside the top group). The SSE falls every round, and
no stump is ever fit to `y` itself -- only to what is still wrong. Check the library:

```python
from sklearn.ensemble import GradientBoostingRegressor

gb_toy = GradientBoostingRegressor(
    n_estimators=3, learning_rate=0.5, max_depth=1,
    random_state=0,
).fit(x_gb, y_gb)
print("sklearn:", gb_toy.predict(x_gb).round(2).tolist())
```

~~~text
sklearn: [2.69, 2.69, 2.69, 10.19, 11.88, 11.88]
~~~

Identical to your round-3 predictions. From here on scikit-learn runs these loops for you,
hundreds of rounds deep, on the coqui data.

## Step 2: The Weak Learner

Boosting's raw material is a *weak learner* -- a model barely better than chance. The
weakest useful tree is a **stump**: one split, depth 1. Fit one and see how poor it is:

```python
stump = DecisionTreeClassifier(max_depth=1, random_state=11).fit(X_train, y_train)

print(f"stump train accuracy: {accuracy_score(y_train, stump.predict(X_train)):.3f}")
print(f"stump test accuracy:  {accuracy_score(y_test, stump.predict(X_test)):.3f}")
```

~~~text
stump train accuracy: 0.683
stump test accuracy:  0.644
~~~

A single yes/no question scores **0.644** on the test set -- a hair above a coin flip.
On its own it is nearly useless. Boosting's claim is that we can chain many of these into
something strong.

## Step 3: AdaBoost Turns Stumps Strong

AdaBoost trains stumps one after another, each time **reweighting the instances the last
stump got wrong** so the next focuses on the hard cases, then combines them by a weighted
vote. Watch the test accuracy climb as the stumps accumulate (current scikit-learn uses
the discrete SAMME algorithm directly):

```python
from sklearn.ensemble import AdaBoostClassifier

for n in (1, 5, 25, 100, 300):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=11),
                             n_estimators=n, learning_rate=0.5,
                             random_state=11).fit(X_train, y_train)
    print(f"n_estimators={n:3d}: test {accuracy_score(y_test, ada.predict(X_test)):.3f}")
```

~~~text
n_estimators=  1: test 0.644
n_estimators=  5: test 0.761
n_estimators= 25: test 0.744
n_estimators=100: test 0.772
n_estimators=300: test 0.783
~~~

One stump scores 0.644 (the same as Step 2); chain 300 of them and the ensemble reaches
**0.783**. The climb is not perfectly smooth -- boosting test curves wobble -- but the
arc is unmistakable: corrective sequencing turned a near-useless learner into a solid
classifier. Each stump is still weak; together, focused on each other's mistakes, they
are strong.

## Step 4: Gradient Boosting and the Learning-Rate Dial -- completion problem

Gradient boosting takes the other route: each new tree is fit to the **residual errors**
of the running ensemble, and `learning_rate` shrinks how much each tree contributes -- the
shrinkage regularization of L10. You saw it at work in Step 1, where a rate of 0.5 made
the second stump repeat the first one's split. A small rate generalizes better but needs
more trees; a large one overfits. Complete the loop to sweep the rate:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Uncomment and complete the marked line:
# for lr in (0.01, 0.1, 0.5, 1.0):
#     gb = GradientBoostingClassifier(n_estimators=200, learning_rate=____, max_depth=2, random_state=11).fit(X_train, y_train)  # set the learning rate
#     tr = accuracy_score(y_train, gb.predict(X_train))
#     te = accuracy_score(y_test, gb.predict(X_test))
#     print(f"learning_rate={lr:<4}: train {tr:.3f}  test {te:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
learning_rate=0.01: train 0.874  test 0.822
learning_rate=0.1 : train 0.974  test 0.811
learning_rate=0.5 : train 1.000  test 0.789
learning_rate=1.0 : train 1.000  test 0.806
~~~

Read the **train** column top to bottom: as the learning rate rises, training accuracy
marches to a perfect **1.000** -- the model is memorizing. Meanwhile the **test** column is
best at the *smallest* rate (0.822 at 0.01) and sags in the middle. The small rate
regularizes: each tree nudges the ensemble a little, so it generalizes instead of
overfitting. The price is needing more trees -- which the next step automates.
</details>

## Step 5: Let Early Stopping Pick the Number of Trees

Rather than guess `n_estimators`, hand gradient boosting a big budget and a validation
slice, and let it stop when validation stops improving (`n_iter_no_change`):

```python
gb_es = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=2,
                                   n_iter_no_change=10, validation_fraction=0.1,
                                   random_state=11).fit(X_train, y_train)

print(f"stopped after {gb_es.n_estimators_} trees")
print(f"train accuracy: {accuracy_score(y_train, gb_es.predict(X_train)):.3f}")
print(f"test accuracy:  {accuracy_score(y_test, gb_es.predict(X_test)):.3f}")
```

~~~text
stopped after 92 trees
train accuracy: 0.919
test accuracy:  0.822
~~~

Given a budget of 1000 trees, it stopped itself at **92** and landed at test **0.822** --
matching the best rate we found by hand in Step 4, with no manual search. Early stopping
is the practical way to size a boosted model: ask for plenty and let validation call it.

## Your Turn

### Exercise 1 -- Bagging versus boosting, head to head

On this same split, fit a `RandomForestClassifier(n_estimators=200)` (bagging, from L12)
and a `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=2)`
(boosting), and print both test accuracies. Which wins -- and by how much?

**Hint:** import `RandomForestClassifier`; fit both with `random_state=11`; print `accuracy_score` on the test set for each.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
random forest (bagging)    test 0.817
gradient boosting (boost)  test 0.811
~~~

They finish neck and neck -- the forest at 0.817, gradient boosting at 0.811. Bagging cut
the single tree's *variance* by averaging; boosting cut its *bias* by sequencing; on this
data both routes arrive at essentially the same place. The forest got there with almost no
tuning, while the boosted model needed a sensible learning rate and tree depth -- the usual
trade: the forest is the easy baseline, boosting the tunable ceiling.
</details>

### Exercise 2 -- Over-boost on purpose

More trees is not always better. Fit a `GradientBoostingClassifier` with `n_estimators=1000`,
`learning_rate=0.5`, `max_depth=3`, and report train and test accuracy. What does the gap
tell you?

**Hint:** one fit with those three settings and `random_state=11`; print train and test accuracy.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
GB n=1000 lr=0.5 depth=3: train 1.000  test 0.800
~~~

A thousand deep-ish trees at a high learning rate drive training accuracy to a perfect
**1.000** while test accuracy slips to **0.800** -- below the early-stopped model's 0.822.
That gap is boosting overfitting: run too long at too large a rate and the ensemble
memorizes the training set, exactly the failure a single deep tree showed in L12. The cure
is the same as Steps 4-5: a smaller learning rate, shallower trees, and early stopping.
</details>

### Exercise 3 -- Round Four, by Hand First

Step 1 stopped after three rounds. Work out round 4 of each algorithm **on paper first**,
then run it to check.

**(a) AdaBoost.** Round 4 picks stump 1 again, `x2 <= 7.5`, which misses only points
4 and 5. Using the weights Step 1 printed after round 3, compute the stump's weighted
error `r`, its vote `alpha`, and point 4's weight after reweighting and renormalizing.
Round 2 had to move away from this stump -- why is it the best choice again now?

**(b) Gradient boosting.** The round-4 stump splits at `x <= 5.5`, so point 6 sits alone
in the right leaf. Starting from point 6's round-3 prediction in the concept note's worked
table (11.875), compute its residual, the right leaf's value, and its new prediction.

Then check both by running one more round in code, and compare (b) with
`GradientBoostingRegressor(n_estimators=4, ...)`.

**Hint:** by hand, `r` is the total weight of the missed points, `alpha = ln((1 - r) / r)`, and a missed weight is multiplied by `e^alpha` before every weight is divided by the new total; a regression leaf predicts the mean residual of its points, and the step is `nu` times that. In code, rerun each Step 1 loop from its starting values.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

By hand:

- **(a)** After round 3, points 4 and 5 weigh 0.0909 each, so
  `r = 0.0909 + 0.0909 = 0.1818` and `alpha = ln(0.8182 / 0.1818) = ln 4.5 = 1.504`.
  Point 4 becomes `0.0909 * 4.5 = 0.4091`. The new total is `0.8182 + 2 * 0.4091 = 1.6364`,
  so point 4 ends at `0.4091 / 1.6364 = 0.25` -- and the two missed points hold half the
  weight again.
- **(b)** Point 6's residual is `14 - 11.875 = 2.125`. It is alone in its leaf, so the leaf
  predicts 2.125, and the new prediction is `11.875 + 0.5 * 2.125 = 12.9375`.

In code:

~~~text
AdaBoost round 4: x2 <= 7.5  missed [4, 5]
  r=0.1818  alpha=1.504
GB round 4: split x <= 5.5  SSE=3.920
  F=[2.475, 2.475, 2.475, 9.975, 11.662, 12.938]
sklearn: [2.475, 2.475, 2.475, 9.975, 11.662, 12.938]
~~~

**(a)** Stump 1 is back because the weights moved. Round 3 missed points 2, 7, 8, and 10,
so they got heavier -- and stump 1 gets all four right. Meanwhile points 4 and 5, the
only points stump 1 misses, fell from 0.25 to 0.0909. AdaBoost can reuse a stump; it
just gets a new vote each time. **(b)** The fourth stump separates point 6, the largest
remaining residual, from the rest; the other five share a leaf of -0.425. The SSE
roughly halves, from 7.984 to 3.920. Each round fixes a smaller piece of what is left,
and scikit-learn agrees to the last digit.
</details>

## Summary

- By hand, AdaBoost reweighted the ten-point toy so the just-missed points always held half
  the weight, and three stumps voted their way to zero errors; gradient boosting fit stumps
  to residuals and cut the SSE from 160 to 7.984 in three half steps. scikit-learn matched
  both loops exactly.
- A single stump scored just 0.644 -- boosting's weak-learner raw material, barely above a
  coin flip.
- AdaBoost reweighted the misclassified instances each round and chained 300 stumps into
  0.783 by a weighted vote -- weak learners made strong by sequencing.
- Gradient boosting fit each tree to the residuals; the learning rate is shrinkage
  regularization -- a small rate (0.01) generalized best (0.822) while a large rate drove
  training accuracy to an overfit 1.000.
- Early stopping sized the model automatically (92 trees, test 0.822), and over-boosting on
  purpose (1000 trees, rate 0.5) overfit to train 1.000 / test 0.800.
- Head to head on identical data, bagging (random forest 0.817) and boosting (0.811)
  finished even -- variance-cut and bias-cut arriving at the same place. This closes Unit IV.
- Next (L14, Unit V): from perceptrons to neural networks -- and Exam I covers Units I-IV.
