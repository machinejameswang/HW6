"""
startup_profit_feature_selection_report.py

Enhanced CRISP-DM report script for the 50 Startups dataset.
It generates one technical chart for each feature-selection algorithm:
1. Pearson Correlation
2. SelectKBest F-Regression
3. Recursive Feature Elimination (RFE)
4. LASSO coefficient path
5. Random Forest feature importance

It also compares model subsets, evaluates the optimized model, and saves deployment-ready artifacts.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_pipeline(numeric_features, include_state=False):
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if include_state:
        transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["State"]))

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("model", LinearRegression())
    ])


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    chart_dir = os.path.join(output_dir, "algorithm_charts")
    os.makedirs(chart_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(script_dir, "50_Startups_dataset.csv"))
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    target = "Profit"
    numeric_features = ["R&D Spend", "Administration", "Marketing Spend"]
    categorical_features = ["State"]

    X = df.drop(columns=[target])
    y = df[target]

    # Data understanding
    corr_profit = df[numeric_features + [target]].corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)
    print("\nCorrelation with Profit:")
    print(corr_profit)

    # Full preprocessing for feature selection algorithms
    pre_all = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ])
    X_all = pre_all.fit_transform(X)
    feature_names = numeric_features + list(pre_all.named_transformers_["cat"].get_feature_names_out(categorical_features))

    # Algorithm 1: Pearson correlation
    pearson_df = pd.DataFrame({"Feature": corr_profit.index, "Correlation": corr_profit.values})
    pearson_df.to_csv(os.path.join(output_dir, "pearson_correlation.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.barh(pearson_df["Feature"], pearson_df["Correlation"])
    plt.axvline(0, linestyle="--")
    plt.gca().invert_yaxis()
    plt.title("Pearson Correlation with Profit")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "01_pearson_correlation.png"), dpi=160)
    plt.close()

    # Algorithm 2: SelectKBest
    skb = SelectKBest(score_func=f_regression, k="all").fit(X_all, y)
    selectk_df = pd.DataFrame({"Feature": feature_names, "F_score": skb.scores_, "p_value": skb.pvalues_}).sort_values("F_score", ascending=False)
    selectk_df.to_csv(os.path.join(output_dir, "selectkbest_scores.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.barh(selectk_df["Feature"], selectk_df["F_score"])
    plt.gca().invert_yaxis()
    plt.title("SelectKBest F-Regression Scores")
    plt.xlabel("F-score")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "02_selectkbest_f_scores.png"), dpi=160)
    plt.close()

    # Algorithm 3: RFE
    rfe = RFE(LinearRegression(), n_features_to_select=2).fit(X_all, y)
    rfe_df = pd.DataFrame({"Feature": feature_names, "RFE_Rank": rfe.ranking_, "Selected": rfe.support_}).sort_values(["RFE_Rank", "Feature"])
    rfe_df.to_csv(os.path.join(output_dir, "rfe_ranking.csv"), index=False)

    plot_df = rfe_df.copy()
    plot_df["Selection_Strength"] = plot_df["RFE_Rank"].max() + 1 - plot_df["RFE_Rank"]
    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["Feature"], plot_df["Selection_Strength"])
    plt.gca().invert_yaxis()
    plt.title("RFE Selection Strength")
    plt.xlabel("Selection strength")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "03_rfe_selection_strength.png"), dpi=160)
    plt.close()

    # Algorithm 4: LASSO
    alphas = np.logspace(1, 5, 90)
    coef_path = []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=50000).fit(X_all, y)
        coef_path.append(lasso.coef_)
    coef_path = np.array(coef_path)

    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=50000).fit(X_all, y)
    lasso_df = pd.DataFrame({"Feature": feature_names, "Coefficient": lasso_cv.coef_})
    lasso_df.to_csv(os.path.join(output_dir, "lasso_coefficients.csv"), index=False)

    plt.figure(figsize=(10, 5))
    for i, name in enumerate(feature_names):
        plt.plot(alphas, coef_path[:, i], label=name)
    plt.xscale("log")
    plt.axhline(0, linestyle="--")
    plt.title("LASSO Coefficient Path")
    plt.xlabel("Alpha")
    plt.ylabel("Coefficient")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "04_lasso_coefficient_path.png"), dpi=160)
    plt.close()

    # Algorithm 5: Random Forest Importance
    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE).fit(X_all, y)
    rf_df = pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    rf_df["Cumulative_Importance"] = rf_df["Importance"].cumsum()
    rf_df.to_csv(os.path.join(output_dir, "random_forest_importance.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(rf_df["Feature"], rf_df["Importance"])
    ax1.set_ylabel("Importance")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(rf_df["Feature"], rf_df["Cumulative_Importance"], marker="o")
    ax2.set_ylabel("Cumulative importance")
    plt.title("Random Forest Importance and Cumulative Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "05_random_forest_importance.png"), dpi=160)
    plt.close()

    # Model subset comparison
    feature_sets = {
        "1: R&D only": (["R&D Spend"], False),
        "2: R&D + Marketing": (["R&D Spend", "Marketing Spend"], False),
        "3: numeric all": (["R&D Spend", "Marketing Spend", "Administration"], False),
        "4: R&D + Marketing + State": (["R&D Spend", "Marketing Spend"], True),
        "5: all predictors": (["R&D Spend", "Marketing Spend", "Administration"], True),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    rows = []
    for name, (nums, include_state) in feature_sets.items():
        pipeline = build_pipeline(nums, include_state)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rows.append({
            "Feature_Set": name,
            "Processed_Feature_Count": len(nums) + (2 if include_state else 0),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": math.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        })
    subset_df = pd.DataFrame(rows)
    subset_df.to_csv(os.path.join(output_dir, "model_subset_comparison.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(subset_df["Processed_Feature_Count"], subset_df["RMSE"], marker="o")
    axes[0].set_title("RMSE by Processed Feature Count")
    axes[0].set_xlabel("Processed feature count")
    axes[0].set_ylabel("RMSE")
    axes[1].plot(subset_df["Processed_Feature_Count"], subset_df["R2"], marker="o")
    axes[1].set_title("R-squared by Processed Feature Count")
    axes[1].set_xlabel("Processed feature count")
    axes[1].set_ylabel("R-squared")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "06_model_subset_curves.png"), dpi=160)
    plt.close()

    # Optimized pipeline
    optimized_pipeline = build_pipeline(["R&D Spend", "Marketing Spend"], include_state=False)
    optimized_pipeline.fit(X_train, y_train)
    y_pred = optimized_pipeline.predict(X_test)

    print("\nOptimized model:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", math.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(optimized_pipeline, X, y, cv=cv, scoring="r2")
    print("CV R2:", cv_scores)
    print("Mean CV R2:", cv_scores.mean())

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=.75, label="Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", label="Perfect prediction")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Actual vs Predicted Profit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "07_actual_vs_predicted.png"), dpi=160)
    plt.close()

    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Profit")
    plt.ylabel("Residual")
    plt.title("Residual Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, "08_residual_analysis.png"), dpi=160)
    plt.close()

    joblib.dump(optimized_pipeline, os.path.join(output_dir, "startup_profit_pipeline.joblib"))
    print("\nSaved output to:", output_dir)


if __name__ == "__main__":
    main()
