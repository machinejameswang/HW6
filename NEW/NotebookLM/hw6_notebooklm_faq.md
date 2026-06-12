# HW6 NotebookLM FAQ

## What is the goal of the project?

The goal is to predict startup Profit using R&D Spend, Administration, Marketing Spend, and State.

## What type of machine learning problem is this?

It is a supervised regression problem because Profit is a continuous numeric target.

## Why is R&D Spend important?

R&D Spend is the strongest predictor across correlation, cross-validation, and feature selection methods.

## Why include Marketing Spend?

Marketing Spend adds secondary business value and gives the model a stronger market-expansion interpretation.

## Why not use all features?

The all-feature model has lower CV R2 Mean than the R&D + Marketing model, so more features do not improve performance here.

## Can the model prove causality?

No. The dataset is observational and has only 50 records, so the results are predictive associations, not causal conclusions.

## What is the final model?

The final main model is `Profit = f(R&D Spend, Marketing Spend)`.

## What is the benchmark model?

The benchmark model is `Profit = f(R&D Spend)`.
