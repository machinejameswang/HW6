# 50 Startups CRISP-DM Feature Selection Technical Report

## 1. Project Summary

This project applies the CRISP-DM methodology to the 50 Startups dataset. The goal is to predict `Profit` and identify which spending-related features provide the strongest predictive signal.

Unlike a simple regression homework, this version performs **algorithm-by-algorithm feature selection analysis** and produces a separate technical chart for each method.

## 2. Dataset Overview

| Item | Result |
|---|---:|
| Records | 50 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Target | Profit |
| Numeric predictors | R&D Spend, Administration, Marketing Spend |
| Categorical predictor | State |

## 3. Algorithm-by-Algorithm Technical Results

### Algorithm 1 - Pearson Correlation

Purpose: measure direct linear relationship between each numeric feature and `Profit`.

| Feature         |   Correlation |
|:----------------|--------------:|
| R&D Spend       |      0.9729   |
| Marketing Spend |      0.747766 |
| Administration  |      0.200717 |

Technical interpretation:

- `R&D Spend` is the strongest direct predictor.
- `Marketing Spend` is useful but weaker than R&D.
- `Administration` has weak direct predictive value.

Chart: `algorithm_charts/01_pearson_correlation.png`

### Algorithm 2 - SelectKBest F-Regression

Purpose: rank features by statistical relationship with the regression target.

| Feature         |    F_score |     p_value |
|:----------------|-----------:|------------:|
| R&D Spend       | 849.789    | 3.50032e-32 |
| Marketing Spend |  60.8815   | 4.38107e-10 |
| Administration  |   2.01496  | 0.162217    |
| State_Florida   |   0.657496 | 0.421448    |
| State_New York  |   0.047275 | 0.828796    |

Technical interpretation:

- The F-score confirms that expenditure-related variables are more important than location features.
- The score gap supports using a compact feature subset.

Chart: `algorithm_charts/02_selectkbest_f_scores.png`

### Algorithm 3 - Recursive Feature Elimination (RFE)

Purpose: recursively remove weak features based on a Linear Regression estimator.

| Feature         |   RFE_Rank | Selected   |
|:----------------|-----------:|:-----------|
| Marketing Spend |          1 | True       |
| R&D Spend       |          1 | True       |
| Administration  |          2 | False      |
| State_Florida   |          3 | False      |
| State_New York  |          4 | False      |

Technical interpretation:

- RFE evaluates features from a model-driven perspective.
- Features ranked 1 are the strongest candidates for the optimized model.

Chart: `algorithm_charts/03_rfe_selection_strength.png`

### Algorithm 4 - LASSO Regression

Purpose: use L1 regularization to shrink weak coefficients toward zero.

Selected alpha from cross-validation:

```text
alpha = 337.6347
```

| Feature         |   Coefficient |   AbsCoefficient |
|:----------------|--------------:|-----------------:|
| R&D Spend       |     36173.3   |        36173.3   |
| Marketing Spend |      3290.92  |         3290.92  |
| Administration  |      -300.143 |          300.143 |
| State_Florida   |         0     |            0     |
| State_New York  |        -0     |            0     |

Technical interpretation:

- LASSO provides embedded feature selection.
- Weak or unstable predictors receive smaller coefficients.
- This is useful when expanding the dataset to higher-dimensional inputs.

Chart: `algorithm_charts/04_lasso_coefficient_path.png`

### Algorithm 5 - Random Forest Feature Importance

Purpose: estimate nonlinear feature contribution using tree-based ensemble learning.

| Feature         |   Importance |   Cumulative_Importance |
|:----------------|-------------:|------------------------:|
| R&D Spend       |   0.92334    |                0.92334  |
| Marketing Spend |   0.0663687  |                0.989709 |
| Administration  |   0.0071917  |                0.9969   |
| State_New York  |   0.0016     |                0.9985   |
| State_Florida   |   0.00149974 |                1        |

Technical interpretation:

- Random Forest confirms that `R&D Spend` dominates prediction.
- Marketing adds secondary value.
- State and Administration appear low-impact relative to spending variables.

Chart: `algorithm_charts/05_random_forest_importance.png`

## 4. Model Subset Curve

The project also compares models with different numbers of processed features.

| Feature_Set                |   Processed_Feature_Count |     MAE |    RMSE |       R2 |
|:---------------------------|--------------------------:|--------:|--------:|---------:|
| 1: R&D only                |                         1 | 6077.36 | 7714.33 | 0.926511 |
| 2: R&D + Marketing         |                         2 | 6469.18 | 8206.33 | 0.916838 |
| 3: numeric all             |                         3 | 6979.15 | 8995.91 | 0.900065 |
| 4: R&D + Marketing + State |                         4 | 6454.51 | 8254.69 | 0.915855 |
| 5: all predictors          |                         5 | 6961.48 | 9055.96 | 0.898727 |

Technical interpretation:

- Adding more features does not automatically improve performance.
- The compact model with `R&D Spend + Marketing Spend` is recommended because it is simpler and competitive.
- The full model can remain as a baseline, but the optimized model is easier to explain and deploy.

Chart: `algorithm_charts/06_model_subset_curves.png`

## 5. Model Evaluation

Optimized model:

```text
Profit = f(R&D Spend, Marketing Spend)
```

| Metric | Value |
|---|---:|
| MAE | 6469.18 |
| RMSE | 8206.33 |
| R-squared | 0.9168 |
| 5-fold CV R-squared mean | 0.9389 |

Residual analysis:

- Residuals are generally centered around zero.
- No strong curve pattern is visible.
- Lower-profit companies still show larger prediction errors.

Charts:

- `algorithm_charts/07_actual_vs_predicted.png`
- `algorithm_charts/08_residual_analysis.png`

## 6. Required Program and Report Modifications

### 6.1 Program changes

1. Add all five feature selection methods:
   - Pearson Correlation
   - SelectKBest F-Regression
   - RFE
   - LASSO
   - Random Forest Importance

2. Save all charts as independent image files.

3. Compare baseline and optimized models separately.

4. Save the final model as a complete pipeline:

```text
startup_profit_pipeline.joblib
```

5. Add 5-fold cross-validation because the dataset has only 50 records.

### 6.2 Report changes

1. Do not only say “Administration was removed because of weak correlation.”
2. Support the decision using multiple algorithms.
3. Explain that `State` is evaluated but has limited incremental value.
4. Explain why the 2-feature optimized model is preferred.
5. Include residual and actual-vs-predicted plots.
6. Present all charts in a one-page infographic.

## 7. Final Technical Conclusion

The final technical conclusion is that startup profitability is primarily explained by R&D investment. Marketing Spend contributes secondary predictive value. Administration and State are not completely meaningless, but they add limited incremental value after the main spending variables are included.

The recommended optimized pipeline is:

```text
Input: R&D Spend + Marketing Spend
Preprocessing: StandardScaler
Model: Linear Regression
Output: Predicted Profit
Deployment artifact: startup_profit_pipeline.joblib
```

This design is simpler, explainable, and ready for Streamlit or API deployment.

## 8. V1/V2 Expert Discussion Update

This update adds a business-facing interpretation layer for the V1 and V2
CRISP-DM scripts.

Updated files:

- `solve_50_startups_crispdm_v1.py`
- `solve_50_startups_crispdm_v2.py`
- `outputs/v1_v2_expert_discussion_final_recommendation.md`
- `outputs/updated_technical_results_summary.csv`
- `outputs/updated_chart_index.csv`

Key update:

- Both V1 and V2 now read `50_Startups_dataset.csv`.
- Both scripts automatically remove the exported index column if present.
- The new expert discussion report compares V1 and V2 through five rounds of
  discussion among a marketing expert, top CEO, regional governor/policy expert,
  and R&D expert.
- The final business recommendation is to use `R&D Spend + Marketing Spend` as
  the main explainable model, while keeping `R&D Spend` alone as the simplest
  benchmark.
- `Administration` and `State` were tested and should not be dismissed without
  evidence, but current technical results show limited incremental value.

Final updated recommendation:

```text
Main model: Profit = f(R&D Spend, Marketing Spend)
Benchmark model: Profit = f(R&D Spend)
Interpretation: predictive association, not causality
```
