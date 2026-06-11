# Final Recommendation - 50 Startups CRISP-DM

## Executive Recommendation

Use `R&D Spend + Marketing Spend` as the final main model:

```text
Profit = f(R&D Spend, Marketing Spend)
```

Keep `R&D Spend` alone as the simplest benchmark model:

```text
Profit = f(R&D Spend)
```

## Why This Is the Best Recommendation

1. `R&D Spend` is the dominant predictor across correlation, SelectKBest, RFE, LASSO, Random Forest, train-test evaluation, and cross-validation.
2. `Marketing Spend` adds only a small CV R2 improvement over R&D alone, but it has strong business meaning because it represents market expansion, brand exposure, and customer acquisition.
3. `Administration` was tested but did not improve performance. It may represent company scale or operational maturity, but it is not useful enough for the final main model.
4. `State` was tested with One-Hot Encoding, but it did not add meaningful predictive value in this 50-row dataset.
5. The full model is not recommended because adding more variables reduced performance and makes interpretation weaker.

## Updated Model Evidence

| Model | Test R2 | Test RMSE | CV R2 Mean | Decision |
|---|---:|---:|---:|---|
| R&D only | 0.9265 | 7714.33 | 0.9374 | Best simple benchmark |
| R&D + Marketing | 0.9168 | 8206.33 | 0.9389 | Final recommended model |
| Numerical Features | 0.9001 | 8995.91 | 0.9338 | Not selected |
| All Features | 0.8987 | 9055.96 | 0.9279 | Not selected |

## Expert Consensus

Marketing expert: R&D creates the product value, while Marketing helps bring that value to the market. The final model should include Marketing for business interpretation, even though the incremental lift is small.

Top CEO: The final model should be simple enough to explain in a budget meeting. `R&D + Marketing` is the best balance between performance, business meaning, and usability.

Governor / Regional Policy Expert: `State` is not selected because the dataset has only 50 rows and limited samples per state. This does not prove regional policy is unimportant; it only means this dataset is too small to support strong regional conclusions.

R&D expert: R&D must remain the core variable. The model results consistently show that product innovation and technical capability are the strongest signals associated with Profit.

## Final Statement

The final recommendation is to use `R&D Spend + Marketing Spend` as the main explainable prediction model and keep `R&D Spend` alone as the benchmark. The results should be described as predictive associations, not causal conclusions, because the dataset is observational and contains only 50 records.

## Updated Files

- `outputs/algorithm_charts/01_pearson_correlation.png`
- `outputs/algorithm_charts/02_selectkbest_f_scores.png`
- `outputs/algorithm_charts/03_rfe_selection_strength.png`
- `outputs/algorithm_charts/04_lasso_coefficient_path.png`
- `outputs/algorithm_charts/05_random_forest_importance.png`
- `outputs/algorithm_charts/06_model_subset_curves.png`
- `outputs/algorithm_charts/07_actual_vs_predicted.png`
- `outputs/algorithm_charts/08_residual_analysis.png`
- `outputs/updated_technical_results_summary.csv`
- `outputs/model_subset_comparison.csv`
- `outputs/startup_profit_pipeline.joblib`

