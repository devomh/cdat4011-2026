---
title: "Lab: Chaining Weak Stumps"
unit: "IV"
lesson: "13"
type: lab
tags: [boosting, adaboost, gradient-boosting, learning-rate, scikit-learn]
difficulty: intermediate
duration: "45 mins"
---

**Goal:** start from a decision stump that barely beats a coin flip, then watch boosting
chain hundreds of them into a classifier that rivals the random forest. You will run
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

## Step 1: The Weak Learner

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

## Step 2: AdaBoost Turns Stumps Strong

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

One stump scores 0.644 (the same as Step 1); chain 300 of them and the ensemble reaches
**0.783**. The climb is not perfectly smooth -- boosting test curves wobble -- but the
arc is unmistakable: corrective sequencing turned a near-useless learner into a solid
classifier. Each stump is still weak; together, focused on each other's mistakes, they
are strong.

## Step 3: Gradient Boosting and the Learning-Rate Dial -- completion problem

Gradient boosting takes the other route: each new tree is fit to the **residual errors**
of the running ensemble, and `learning_rate` shrinks how much each tree contributes -- the
shrinkage regularization of L10. A small rate generalizes better but needs more trees; a
large one overfits. Complete the loop to sweep the rate:

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

## Step 4: Let Early Stopping Pick the Number of Trees

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
matching the best rate we found by hand in Step 3, with no manual search. Early stopping
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
is the same as Step 3-4: a smaller learning rate, shallower trees, and early stopping.
</details>

## Summary

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
