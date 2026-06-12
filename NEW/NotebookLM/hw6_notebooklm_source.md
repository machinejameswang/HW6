# HW6 NotebookLM Source

This source summarizes the HW6 Kaggle 50 Startups CRISP-DM project for NotebookLM.

## Core Conclusion

Use `R&D Spend + Marketing Spend` as the main explainable prediction model.
Keep `R&D Spend` alone as the benchmark model.
Interpret the results as predictive associations, not causal conclusions.

## Slide 1: HW6 Project Overview

### Key facts

- Project: Kaggle 50 Startups Profit Prediction
- Methodology: CRISP-DM
- Library: Scikit-learn
- Task type: supervised regression

### Why it matters

This slide introduces the project scope and frames the work as a complete machine learning workflow.

### Narration

This project predicts startup profit using the Kaggle 50 Startups dataset. It follows CRISP-DM and uses Scikit-learn regression models.

## Slide 2: Business Objective

### Key facts

- Goal: predict Profit from spending variables
- Use case: resource allocation and scenario planning
- Audience: founders, investors, analysts, and managers
- Interpretation: predictive association, not causality

### Why it matters

The model supports better business discussion about limited startup resources.

### Narration

The business objective is to estimate profit from startup spending patterns. The result supports decision-making but should not be interpreted as causal proof.

## Slide 3: Dataset Structure

### Key facts

- File: 50_Startups_dataset.csv
- Rows: 50
- Target: Profit
- Features: R&D Spend, Administration, Marketing Spend, State

### Why it matters

The dataset is small, so cross-validation is important.

### Narration

The dataset has 50 records and five useful columns. Profit is the target variable, and the remaining columns are predictors.

## Slide 4: CRISP-DM Six Steps

### Key facts

- 1. Business Understanding
- 2. Data Understanding
- 3. Data Preparation
- 4. Modeling
- 5. Evaluation
- 6. Deployment

### Why it matters

CRISP-DM connects business questions, data analysis, modeling, evaluation, and deployment.

### Narration

The workflow follows the six steps of CRISP-DM, giving the project a clear structure from problem definition to deployment simulation.

## Slide 5: Data Quality

### Key facts

- Dataset shape after cleaning: 50 rows, 5 columns
- Missing values: 0
- Duplicate rows: 0
- States: New York, California, Florida

### Why it matters

Clean input data makes the modeling workflow simpler, but the small sample size remains a limitation.

### Narration

The dataset has no missing values and no duplicate rows. The state categories are New York, California, and Florida.

## Slide 6: Feature Interpretation

### Key facts

- R&D Spend: product innovation and technical capability
- Marketing Spend: market expansion and customer acquisition
- Administration: operating scale and management structure
- State: regional auxiliary factor

### Why it matters

Feature meaning helps explain why a model is useful for business decisions.

### Narration

Each feature has a business meaning. R&D is linked to innovation, Marketing to market reach, Administration to operating scale, and State to regional context.

## Slide 7: Correlation Evidence

### Key facts

- R&D Spend correlation with Profit: 0.9729
- Marketing Spend correlation with Profit: 0.7478
- Administration correlation with Profit: 0.2007
- R&D Spend has the strongest direct linear association

### Why it matters

Correlation gives an initial signal about which numeric features are most associated with Profit.

### Narration

R&D Spend has the strongest correlation with Profit. Marketing is also positive but weaker, while Administration is much weaker.

## Slide 8: Data Preparation

### Key facts

- Separate X features and y target
- Use Profit as target
- Encode State using OneHotEncoder
- Use ColumnTransformer and Pipeline
- Use train_test_split with test_size=0.2 and random_state=42

### Why it matters

A Scikit-learn Pipeline keeps preprocessing and modeling reproducible.

### Narration

The preparation step separates features and target, encodes the categorical State variable, and uses a Pipeline for reproducibility.

## Slide 9: Modeling Strategy

### Key facts

- Primary model: LinearRegression
- Reason: target is continuous
- Reason: dataset is small
- Reason: model is interpretable

### Why it matters

Linear Regression is simple, explainable, and appropriate for a teaching-focused regression project.

### Narration

Linear Regression was selected because it is interpretable and suitable for a small regression dataset.

## Slide 10: Model Experiments

### Key facts

- Model 1: R&D only
- Model 2: R&D + Marketing
- Model 3: numerical features
- Model 4: all features

### Why it matters

Comparing feature sets tests whether more variables actually improve predictive performance.

### Narration

The project compares four models to test the value of R&D, Marketing, Administration, and State.

## Slide 11: Evaluation Metrics

### Key facts

- Train-test metrics: R2, MAE, RMSE
- Cross-validation: 5-fold KFold
- CV metrics: CV R2 Mean, CV R2 Std, CV RMSE Mean, CV RMSE Std
- Small dataset requires cross-validation

### Why it matters

Cross-validation reduces reliance on a single train-test split.

### Narration

The models are evaluated using both train-test metrics and 5-fold cross-validation because the dataset has only 50 rows.

## Slide 12: Model Comparison Results

### Key facts

- R&D only: Test R2 0.9265, CV R2 Mean 0.9374
- R&D + Marketing: Test R2 0.9168, CV R2 Mean 0.9389
- Numerical Features: Test R2 0.9001, CV R2 Mean 0.9338
- All Features: Test R2 0.8987, CV R2 Mean 0.9279

### Why it matters

R&D + Marketing has the best CV R2 Mean, while the full model performs worse.

### Narration

The model comparison shows that R&D plus Marketing has the highest cross-validation R2 mean, while the all-feature model performs worse.

## Slide 13: Feature Selection Methods

### Key facts

- Pearson correlation
- SelectKBest F-regression
- Recursive Feature Elimination
- LASSO
- Random Forest importance

### Why it matters

Using several methods makes the feature conclusion more robust.

### Narration

Several feature selection methods were used to avoid relying on only one technique.

## Slide 14: Feature Selection Consensus

### Key facts

- R&D Spend is dominant
- Marketing Spend is secondary
- Administration has limited incremental value
- State has very limited value in this small dataset

### Why it matters

The same conclusion appears across statistical, model-based, regularized, and tree-based methods.

### Narration

The feature selection methods consistently identify R&D Spend as the strongest feature, with Marketing as a secondary feature.

## Slide 15: Final Model Recommendation

### Key facts

- Main model: Profit = f(R&D Spend, Marketing Spend)
- Benchmark model: Profit = f(R&D Spend)
- Do not choose the most complex model automatically
- Prefer performance, stability, simplicity, and interpretability

### Why it matters

The final model balances predictive performance with business meaning.

### Narration

The recommended main model uses R&D Spend and Marketing Spend. R&D alone remains the benchmark.

## Slide 16: Deployment Simulation

### Key facts

- Sample R&D Spend: 120,000
- Sample Administration: 130,000
- Sample Marketing Spend: 250,000
- Sample State: New York
- Predicted Profit: 150,042.94

### Why it matters

The deployment step demonstrates how the final pipeline can make a prediction for a new startup.

### Narration

The deployment simulation predicts profit for a sample startup and saves the model pipeline for reuse.

## Slide 17: Expert Consensus

### Key facts

- Marketing expert: Marketing supports market reach
- Top CEO: model must be explainable in business meetings
- Policy expert: State needs more data before strong claims
- R&D expert: R&D is the core innovation signal

### Why it matters

The expert panel makes the model interpretation more realistic and business-facing.

### Narration

The expert panel agrees that R&D is the core signal, Marketing adds business context, and State should not be overinterpreted.

## Slide 18: Limitations

### Key facts

- Only 50 observations
- Observational data
- Possible multicollinearity between R&D and Marketing
- State has limited samples per category
- Results are predictive associations, not causal conclusions

### Why it matters

Limitations prevent overclaiming and guide future improvements.

### Narration

The dataset is small and observational, so the results should be interpreted carefully.

## Slide 19: Artifacts Produced

### Key facts

- V1 and V2 CRISP-DM Python scripts
- Technical charts and CSV summaries
- Final model pipeline
- 20-slide PPT with notes and male narration
- HeyGen and Hyperframes packages

### Why it matters

The project includes code, reports, visuals, models, and presentation materials.

### Narration

The final project includes scripts, charts, summaries, models, and presentation files.

## Slide 20: Final Conclusion

### Key facts

- R&D Spend is the strongest predictor
- Marketing Spend adds secondary business value
- Administration and State add limited value here
- Use R&D + Marketing as the main model
- Keep R&D only as the benchmark

### Why it matters

This conclusion is supported by model evaluation and feature selection evidence.

### Narration

The final conclusion is to use R&D Spend plus Marketing Spend as the main model, while keeping R&D Spend alone as the benchmark.
