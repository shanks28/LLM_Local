# ML vs AI: A Quick Guide

## Overview
- **AI**: Systems that perform tasks typically requiring human intelligence (perception, reasoning, decision‑making, language).
- **ML**: A subset of AI where models learn patterns from data to improve performance on a task without explicit rule‑coding.

## Core ML Concepts
- **Data**: Inputs and ground‑truth labels (for supervised tasks).
- **Features**: Measurable attributes used by models; may be engineered or learned.
- **Model**: A function with parameters (e.g., linear regression, tree, neural net).
- **Training**: Optimizing parameters to minimize a loss on training data.
- **Validation/Test**: Measuring generalization on held‑out data using metrics.

## ML Types
- **Supervised learning**: Train from labeled data (classification, regression).
- **Unsupervised learning**: Find structure without labels (clustering, dimensionality reduction).
- **Reinforcement learning**: Learn actions via rewards in an environment.

## Typical ML Pipeline
1. Problem framing and success metrics
2. Data collection and splitting (train/validation/test)
3. Feature engineering or representation learning
4. Model selection and training
5. Evaluation (metrics, error analysis)
6. Iteration and tuning (regularization, hyperparameters)
7. Deployment and monitoring (drift, retraining)

## Common Algorithms
- Classification: logistic regression, decision trees/random forests, SVMs, neural nets
- Regression: linear/ridge/lasso regression, gradient boosting, neural nets
- Unsupervised: k‑means, DBSCAN, PCA, t‑SNE/UMAP, autoencoders

## Key Differences (AI vs ML)
- Scope: AI is the broader field; ML is a technique within AI.
- Approach: AI can be rule‑based; ML is data‑driven learning.
- Maintenance: Rules can be brittle; ML adapts with data and retraining.

## Glossary
- **Loss function**: Quantifies prediction error to guide training.
- **Overfitting**: Memorizing training data; poor generalization.
- **Regularization**: Techniques to reduce overfitting.
- **Metric**: Task‑specific measure (accuracy, F1, MAE, AUC).
