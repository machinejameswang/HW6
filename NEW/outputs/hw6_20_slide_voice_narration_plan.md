# HW6 20-Page PPT Plan with Voice Narration

Topic: Kaggle 50 Startups CRISP-DM Scikit-learn Project  
Format: 20 slides with built-in voice narration script  
Style suggestion: clean business/data-science presentation, light background, blue/green/orange accents

---

## Slide 1 - Title

### On-slide content

**HW6: Kaggle 50 Startups Profit Prediction**

CRISP-DM + Scikit-learn Regression Project

Prepared by: James Wang

### Visual suggestion

Use the Excalidraw-style infographic or a simple startup + analytics icon.

### Voice-over script

Welcome to my HW6 project presentation. In this project, I use the Kaggle 50 Startups dataset to predict startup profit. The analysis follows the CRISP-DM process and uses Scikit-learn regression models to compare different feature sets.

---

## Slide 2 - Project Objective

### On-slide content

**Goal**

- Predict startup `Profit`
- Use business spending variables
- Build an interpretable regression model
- Support business decision-making

### Visual suggestion

Show a simple arrow from business spending to predicted profit.

### Voice-over script

The main objective is to predict startup profit based on spending information. The model is designed to be interpretable, so it can support decisions by founders, investors, analysts, and managers.

---

## Slide 3 - Dataset Overview

### On-slide content

**Dataset: 50_Startups_dataset.csv**

- Records: 50
- Target: `Profit`
- Features: `R&D Spend`, `Administration`, `Marketing Spend`, `State`
- Problem type: supervised regression

### Visual suggestion

Use a small table with columns: Feature, Type, Role.

### Voice-over script

The dataset contains 50 startup records. The target variable is Profit. The predictors include R&D spending, administrative spending, marketing spending, and state. Since Profit is a continuous numeric value, this is a supervised regression problem.

---

## Slide 4 - CRISP-DM Workflow

### On-slide content

**Six CRISP-DM Steps**

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

### Visual suggestion

Use the draw.io workflow or Mermaid flow preview from `hw6.md`.

### Voice-over script

This project follows the six steps of CRISP-DM. This structure helps connect the business question, data exploration, model building, evaluation, and final deployment simulation into one complete workflow.

---

## Slide 5 - Business Understanding

### On-slide content

**Business Question**

Which spending patterns are associated with higher startup profit?

**Important note**

Use prediction language, not causal language.

### Visual suggestion

Show four stakeholders: founder, investor, manager, analyst.

### Voice-over script

The business question is which spending patterns are associated with higher profit. Because the dataset is observational and small, I avoid claiming causality. The model can identify predictive associations, but it cannot prove that one feature directly causes profit to increase.

---

## Slide 6 - Expert Panel View

### On-slide content

**Expert Perspectives**

- Marketing expert: market expansion
- Top CEO: decision usefulness
- Governor / policy expert: regional factors
- R&D expert: product innovation

### Visual suggestion

Use four small expert cards.

### Voice-over script

To make the interpretation more business-oriented, I use four expert perspectives. The marketing expert focuses on customer acquisition, the CEO focuses on decision usefulness, the policy expert considers regional factors, and the R&D expert focuses on product innovation.

---

## Slide 7 - Data Understanding

### On-slide content

**Data Quality**

- Shape: 50 rows, 5 columns
- Missing values: 0
- Duplicate rows: 0
- States: New York, California, Florida

### Visual suggestion

Use checklist icons for missing values and duplicates.

### Voice-over script

The dataset is clean. It has 50 rows and 5 useful columns after removing the exported index column. There are no missing values and no duplicate rows. The state variable contains New York, California, and Florida.

---

## Slide 8 - Feature Meaning

### On-slide content

**Feature Roles**

- `R&D Spend`: product innovation
- `Marketing Spend`: market expansion
- `Administration`: operating scale
- `State`: regional auxiliary factor

### Visual suggestion

Use four icons: lab, megaphone, office, map pin.

### Voice-over script

Each feature has a business meaning. R&D spending reflects product development and innovation. Marketing spending reflects market expansion and customer acquisition. Administration may represent operating scale. State may represent regional conditions, but the sample size is too small for strong regional conclusions.

---

## Slide 9 - Correlation Results

### On-slide content

**Correlation with Profit**

- R&D Spend: 0.9729
- Marketing Spend: 0.7478
- Administration: 0.2007

### Visual suggestion

Use `01_pearson_correlation.png`.

### Voice-over script

The correlation results show that R&D Spend has the strongest direct linear relationship with Profit. Marketing Spend is also positively associated with Profit, but it is weaker than R&D. Administration has a much weaker correlation.

---

## Slide 10 - Data Preparation

### On-slide content

**Preparation Pipeline**

- Separate `X` and `y`
- Encode `State` with OneHotEncoder
- Use ColumnTransformer
- Use Scikit-learn Pipeline
- Train-test split: 80/20

### Visual suggestion

Show a pipeline diagram: raw data -> preprocessing -> model.

### Voice-over script

For data preparation, I separate the features and target variable. The categorical State feature is encoded with OneHotEncoder, not label encoding. I use ColumnTransformer and Pipeline to keep preprocessing and modeling in one reproducible workflow.

---

## Slide 11 - Modeling Strategy

### On-slide content

**Primary Model**

LinearRegression

**Why**

- Continuous target
- Small dataset
- Easy to explain
- Suitable for CRISP-DM teaching

### Visual suggestion

Use a simple regression line graphic.

### Voice-over script

The primary model is Linear Regression. This model is appropriate because the target is continuous, the dataset is small, and the results are easier to explain in a business and educational setting.

---

## Slide 12 - Model Experiments

### On-slide content

**Four Feature Sets**

1. R&D only
2. R&D + Marketing
3. Numerical features
4. All features

### Visual suggestion

Use four stacked model cards.

### Voice-over script

I compare four feature sets. First, I test R&D alone. Second, I add Marketing. Third, I include all numerical features. Finally, I include all features, including State. This helps test whether additional variables actually improve prediction.

---

## Slide 13 - Evaluation Metrics

### On-slide content

**Metrics**

- R2 Score
- MAE
- RMSE
- CV R2 Mean
- CV R2 Std
- CV RMSE Mean

### Visual suggestion

Use a dashboard-style metrics panel.

### Voice-over script

The models are evaluated with R2, MAE, and RMSE on the train-test split. Since the dataset has only 50 rows, I also use 5-fold cross-validation to check model stability and reduce reliance on one split.

---

## Slide 14 - Model Comparison

### On-slide content

| Model | Test R2 | CV R2 Mean |
|---|---:|---:|
| R&D only | 0.9265 | 0.9374 |
| R&D + Marketing | 0.9168 | 0.9389 |
| Numerical | 0.9001 | 0.9338 |
| All features | 0.8987 | 0.9279 |

### Visual suggestion

Use a clean table with highlighted final model row.

### Voice-over script

The comparison shows that R&D only performs very strongly. However, R&D plus Marketing has the highest cross-validation R2 mean, at 0.9389. The numerical and all-feature models perform worse, which shows that more features do not automatically improve the model.

---

## Slide 15 - Feature Selection Evidence

### On-slide content

**Feature Selection Methods**

- Pearson correlation
- SelectKBest
- RFE
- LASSO
- Random Forest importance

**Consensus**

R&D Spend is dominant.

### Visual suggestion

Use a five-method evidence funnel.

### Voice-over script

Multiple feature selection methods support the same conclusion. Pearson correlation, SelectKBest, RFE, LASSO, and Random Forest importance all show that R&D Spend is the dominant predictive feature.

---

## Slide 16 - Random Forest Importance

### On-slide content

**Importance Ranking**

- R&D Spend: 0.9228
- Marketing Spend: 0.0670
- Administration: 0.0072
- State total: about 0.0030

### Visual suggestion

Use `05_random_forest_importance.png`.

### Voice-over script

The Random Forest feature importance result also confirms the same pattern. R&D Spend contributes most of the predictive signal. Marketing is second, while Administration and State contribute very little in this dataset.

---

## Slide 17 - Final Model Decision

### On-slide content

**Final Main Model**

`Profit = f(R&D Spend, Marketing Spend)`

**Benchmark**

`Profit = f(R&D Spend)`

### Visual suggestion

Use a decision box with a check mark on R&D + Marketing.

### Voice-over script

The final main model uses R&D Spend and Marketing Spend. R&D alone is kept as the simplest benchmark. This choice balances predictive performance, business meaning, simplicity, and interpretability.

---

## Slide 18 - Deployment Simulation

### On-slide content

**Sample Input**

- R&D Spend: 120,000
- Administration: 130,000
- Marketing Spend: 250,000
- State: New York

**Predicted Profit**

150,042.94

### Visual suggestion

Use a small input-output prediction card.

### Voice-over script

For deployment simulation, I use a sample startup with R&D Spend of 120,000, Administration of 130,000, Marketing Spend of 250,000, and State as New York. The final model predicts a profit of about 150,042.94.

---

## Slide 19 - Business Interpretation

### On-slide content

**Business Meaning**

- Prioritize product innovation
- Use Marketing to support market reach
- Do not overvalue Administration or State
- Avoid causal claims

### Visual suggestion

Use a simple strategy pyramid: R&D at the base, Marketing above.

### Voice-over script

The business interpretation is that product innovation appears to be the strongest signal associated with profit. Marketing can support market reach, but its incremental value is small. Administration and State were tested, but they do not add meaningful predictive value in this small dataset.

---

## Slide 20 - Final Conclusion

### On-slide content

**Final Conclusion**

Use `R&D Spend + Marketing Spend` as the main explainable model.

Keep `R&D Spend` as the benchmark.

Interpret results as predictive associations, not causality.

### Visual suggestion

Use the Excalidraw A4 infographic as the closing visual.

### Voice-over script

In conclusion, the recommended model is R&D Spend plus Marketing Spend. R&D Spend remains the strongest predictor, while Marketing provides useful business context. Because the dataset contains only 50 observations, the results should be treated as predictive associations rather than causal conclusions. Thank you.

