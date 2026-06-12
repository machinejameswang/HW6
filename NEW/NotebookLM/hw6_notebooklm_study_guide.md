# HW6 NotebookLM Study Guide

## One-sentence summary

This CRISP-DM project predicts startup Profit and concludes that R&D Spend is the dominant predictor, with Marketing Spend as a useful secondary feature.

## Must-know concepts

- CRISP-DM
- Supervised regression
- Linear Regression
- Train-test split
- 5-fold cross-validation
- OneHotEncoder
- ColumnTransformer
- Pipeline
- R2, MAE, RMSE
- Feature selection

## Final model

```text
Profit = f(R&D Spend, Marketing Spend)
```

## Benchmark model

```text
Profit = f(R&D Spend)
```

## Key numbers

- R&D only CV R2 Mean: 0.9374
- R&D + Marketing CV R2 Mean: 0.9389
- All features CV R2 Mean: 0.9279
- Sample predicted Profit: 150,042.94

## Main caution

The dataset has only 50 rows, so use cross-validation and avoid causal claims.
