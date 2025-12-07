# Random Forest: Simple Overview

## What It Is
- Combines the outputs of many decision trees to produce a single prediction.
- Easy to use and flexible; handles both classification and regression tasks.

## Decision Trees (Building Blocks)
- A tree asks a sequence of questions about features.
- Each path ends at a leaf node that gives the final decision/prediction.

## Ensemble Methods
- Common types: bagging (bootstrap aggregation) and boosting.
- Bagging: train each tree on a random sample of the training data drawn with replacement (a point can appear multiple times). This reduces variance in noisy datasets.
- Boosting (for context): trains trees sequentially, focusing on previous errors to reduce bias.

## How Random Forest Works
- Mixes bagging with feature randomness.
- At each tree split, it considers only a random subset of features, making trees less correlated.
- Lower correlation across trees leads to better generalization and more precise predictions.

## Random Forest vs. Single Decision Tree
- Single tree: considers all features at each split.
- Random forest: considers a random subset of features at each split, creating diverse trees.

## Key Hyperparameters
- `node_size` (minimum samples per leaf or related stopping criteria)
- `number of trees` (e.g., `n_estimators`)
- `number of features sampled` at each split (e.g., `max_features`)

## Tasks It Solves
- Classification: predict discrete classes (e.g., approve vs reject).
- Regression: predict a continuous number (e.g., price, risk score).

## Benefits
- Reduced risk of overfitting through averaging.
- Flexibility across different data types and tasks.

## Simple Regression Example
- Given input `A` with output `B`, estimate the output at a nearby point `A + x`.
- Typical questions: “how much?” or “how many?” when the answer is a number (continuous).
