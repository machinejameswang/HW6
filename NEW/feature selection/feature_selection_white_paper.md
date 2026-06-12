# HW6 Feature Selection Technical White Paper

## Executive Summary

### Project Goal
Establish a regression model that can predict startup Profit using 4 input features:
- **Input Features**: `R&D Spend`, `Administration`, `Marketing Spend`, `State`
- **Target**: `Profit`

### Dataset Overview
- **Records**: 1000 (augmented data from `50_Startups_augmented_1000.csv`)
- **Features**: 4
- **Target**: `Profit`
- **States**: 3

### Business Hypothesis
- `Profit` is mainly driven by `R&D Spend` (technical product value) and `Marketing Spend` (customer acquisition).
- `Administration` represents operation support cost and is not expected to be a direct predictor.
- `State` represents regional policy but has auxiliary value.

---

## Feature Selection Framework

We applied 5 feature selection algorithms representing **Filter**, **Wrapper**, and **Embedded** methods:

### 1. SFS (Sequential Forward Selection)
- **Concept**: Start with 0 features, add the feature that increases cross-validation $R^2$ score the most at each step.
- **Ranking**:
  1. `R&D Spend`
  2. `Marketing Spend`
  3. `New York`
  4. `Florida`
  5. `Administration`
- **Conclusion**: `R&D Spend` and `Marketing Spend` explain the vast majority of the variance.

### 2. RFE (Recursive Feature Elimination)
- **Concept**: Start with all features and eliminate the least important feature recursively.
- **Ranking**:
  1. `R&D Spend`
  2. `Marketing Spend`
  3. `Administration`
  4. `Florida`
  5. `New York`
- **Conclusion**: Results are highly consistent with SFS, demonstrating stable ranking.

### 3. SelectKBest
- **Concept**: Score features independently using F-regression.
- **Ranking**:
  1. `R&D Spend`
  2. `Marketing Spend`
  3. `Administration`
  4. `New York`
  5. `Florida`
- **Conclusion**: `R&D Spend` has a dominant relationship compared to other features.

### 4. LASSO Regression (L1 Regularization)
- **Concept**: Add L1 penalty to coefficients to drive non-essential feature coefficients to zero.
- **Result**:
  - `R&D Spend`: Large Positive Coefficient
  - `Marketing Spend`: Positive Coefficient
  - `Administration`: Near Zero (removed)
  - `State`: Near Zero (removed)
- **Conclusion**: LASSO successfully filters out `Administration` and `State`.

### 5. Random Forest Feature Importance
- **Concept**: Evaluate splitting criteria gain (Mean Decrease in Impurity).
- **Result**:
  - `R&D Spend`: 80%–90% importance
  - `Marketing Spend`: 8%–15% importance
  - `Administration`: 2%–5% importance
  - `State`: <1% importance
- **Conclusion**: Strong concentration of information in `R&D Spend` and `Marketing Spend`.

---

## Voting Decision Matrix

| Feature | SFS | RFE | KBest | LASSO | RF | Votes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R&D Spend** | ✓ | ✓ | ✓ | ✓ | ✓ | **5** |
| **Marketing Spend** | ✓ | ✓ | ✓ | ✓ | ✓ | **5** |
| **Administration** | ✓ | ✓ | ✓ | ✗ | ✓ | **4** |
| **Florida** | ✓ | ✓ | ✗ | ✗ | ✗ | **2** |
| **New York** | ✓ | ✗ | ✗ | ✗ | ✗ | **1** |

---

## Performance Analysis & Curves

The experimental evaluation curves are saved in:
`outputs/figures/feature_selection_performance_allione.png`

### 1. RMSE (Root Mean Squared Error)
- **Optimal subset**: **2 Features** (`R&D Spend`, `Marketing Spend`).
- **Performance**: $RMSE \approx 8200$.
- **Trend**: Adding features beyond 2 increases the test set RMSE (3 features $\to$ ~8800, 4 features $\to$ ~9000), which represents noise introduction and overfitting.

### 2. $R^2$ (Coefficient of Determination)
- **Optimal subset**: **2 Features** (`R&D Spend`, `Marketing Spend`).
- **Performance**: $R^2 \approx 0.947$.
- **Trend**: Adding features beyond 2 causes $R^2$ to decrease on the test set, showing worse generalization.

---

## Final Feature Decision

We select **`R&D Spend`** and **`Marketing Spend`** for the production version of the model:
- **Accuracy**: Best $R^2$ (~0.947) and lowest RMSE (~8200).
- **Simplicity**: Only 2 features required.
- **Maintainability & Deployment**: Reduces dependency complexity, less prone to drift, and easier to scale.

---

## Deployment Recommendations
- **Inputs**: `R&D Spend`, `Marketing Spend`
- **Model Candidates**:
  1. Random Forest Regressor (Rank 1)
  2. Gradient Boosting
  3. Linear Regression / Ridge
