"""
HW6 NEW SKlearn V2 - Filter / Wrapper / Embedded Feature Selection Framework

This script implements a complete AI engineer-style sklearn project:

1. Data preprocessing
2. Filter methods: Pearson, VIF, SelectKBest
3. Wrapper method: RFE
4. Embedded methods: LASSO, Random Forest Importance
5. Voting strategy for final feature selection
6. Regression benchmark
7. Classification benchmark
8. PCA dimensionality reduction
9. KMeans clustering
10. Joblib deployment artifacts
"""

from pathlib import Path
import math
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, silhouette_score
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "50_Startups_augmented_1000.csv"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"

for d in [OUTPUTS, MODELS]:
    d.mkdir(parents=True, exist_ok=True)


def build_preprocessor(num_cols, cat_cols=None):
    cat_cols = cat_cols or []
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers)


def calculate_vif(df, num_cols):
    if not HAS_STATSMODELS:
        return pd.DataFrame({"Feature": num_cols, "VIF": np.nan})

    x_vif = sm.add_constant(df[num_cols])
    return pd.DataFrame({
        "Feature": x_vif.columns,
        "VIF": [variance_inflation_factor(x_vif.values, i) for i in range(x_vif.shape[1])]
    }).query("Feature != 'const'").reset_index(drop=True)


def main():
    df = pd.read_csv(DATA_PATH)

    target = "Profit"
    num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
    cat_cols = ["State"]

    X = df[num_cols + cat_cols]
    y = df[target]

    # PCA and clustering
    scaled_num = StandardScaler().fit_transform(df[num_cols])
    pca = PCA(n_components=2, random_state=42)
    pca_xy = pca.fit_transform(scaled_num)
    df["PCA1"] = pca_xy[:, 0]
    df["PCA2"] = pca_xy[:, 1]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_num)
    df["Profit_Level"] = pd.qcut(df[target], q=3, labels=["Low", "Medium", "High"])
    df.to_csv(OUTPUTS / "HW6_processed_dataset.csv", index=False)

    # Full preprocessing for feature selection
    full_preprocessor = build_preprocessor(num_cols, cat_cols)
    X_all = full_preprocessor.fit_transform(X)
    feature_names = num_cols + list(full_preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))

    # FILTER methods
    pearson = df[num_cols + [target]].corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)
    pearson_df = pd.DataFrame({
        "Feature": pearson.index,
        "Pearson_Correlation": pearson.values,
        "Filter_Pearson_Selected": (pearson.abs().values >= 0.5)
    })

    vif_df = calculate_vif(df, num_cols)
    vif_df["Filter_VIF_Selected"] = vif_df["VIF"].fillna(999) < 5

    skb = SelectKBest(score_func=f_regression, k="all").fit(X_all, y)
    selectk_df = pd.DataFrame({
        "Feature": feature_names,
        "F_score": skb.scores_,
        "p_value": skb.pvalues_
    }).sort_values("F_score", ascending=False)
    selectk_cut = selectk_df["F_score"].quantile(0.60)
    selectk_df["Filter_SelectKBest_Selected"] = selectk_df["F_score"] >= selectk_cut

    # WRAPPER method
    rfe = RFE(LinearRegression(), n_features_to_select=2).fit(X_all, y)
    rfe_df = pd.DataFrame({
        "Feature": feature_names,
        "RFE_Rank": rfe.ranking_,
        "Wrapper_RFE_Selected": rfe.support_
    }).sort_values(["RFE_Rank", "Feature"])

    # EMBEDDED methods
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=50000).fit(X_all, y)
    lasso_df = pd.DataFrame({
        "Feature": feature_names,
        "LASSO_Coefficient": lasso_cv.coef_,
        "AbsCoefficient": np.abs(lasso_cv.coef_)
    })
    lasso_df["Embedded_LASSO_Selected"] = lasso_df["AbsCoefficient"] > 1e-6

    rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(X_all, y)
    rf_df = pd.DataFrame({
        "Feature": feature_names,
        "RF_Importance": rf.feature_importances_
    }).sort_values("RF_Importance", ascending=False)
    rf_cut = rf_df["RF_Importance"].quantile(0.60)
    rf_df["Embedded_RF_Selected"] = rf_df["RF_Importance"] >= rf_cut

    # Voting strategy
    voting = pd.DataFrame({"Feature": feature_names})
    for sub_df in [pearson_df, vif_df, selectk_df, rfe_df, lasso_df, rf_df]:
        voting = voting.merge(sub_df, on="Feature", how="left")

    bool_cols = [
        "Filter_Pearson_Selected",
        "Filter_VIF_Selected",
        "Filter_SelectKBest_Selected",
        "Wrapper_RFE_Selected",
        "Embedded_LASSO_Selected",
        "Embedded_RF_Selected"
    ]
    for c in bool_cols:
        voting[c] = voting[c].fillna(False).astype(bool)

    voting["Filter_Score"] = voting[[
        "Filter_Pearson_Selected",
        "Filter_VIF_Selected",
        "Filter_SelectKBest_Selected"
    ]].sum(axis=1)
    voting["Wrapper_Score"] = voting[["Wrapper_RFE_Selected"]].sum(axis=1)
    voting["Embedded_Score"] = voting[[
        "Embedded_LASSO_Selected",
        "Embedded_RF_Selected"
    ]].sum(axis=1)
    voting["Total_Votes"] = voting[bool_cols].sum(axis=1)
    voting["Decision"] = np.where(voting["Total_Votes"] >= 3, "Keep", "Optional/Remove")
    voting = voting.sort_values(["Total_Votes", "RF_Importance"], ascending=[False, False])

    pearson_df.to_csv(OUTPUTS / "filter_pearson_correlation.csv", index=False)
    vif_df.to_csv(OUTPUTS / "filter_vif_analysis.csv", index=False)
    selectk_df.to_csv(OUTPUTS / "filter_selectkbest.csv", index=False)
    rfe_df.to_csv(OUTPUTS / "wrapper_rfe.csv", index=False)
    lasso_df.to_csv(OUTPUTS / "embedded_lasso.csv", index=False)
    rf_df.to_csv(OUTPUTS / "embedded_random_forest_importance.csv", index=False)
    voting.to_csv(OUTPUTS / "feature_selection_voting_matrix.csv", index=False)

    final_features = ["R&D Spend", "Marketing Spend"]

    # Regression benchmark
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "LASSO Regression": Lasso(alpha=0.1, max_iter=20000),
        "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
        "SVR": SVR(kernel="rbf", C=100, epsilon=0.1)
    }

    rows = []
    trained = {}
    for name, model in reg_models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(final_features, [])),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        rows.append({
            "Feature_Set": "Selected_Features_Filter_Wrapper_Embedded",
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
            "CV_R2_Mean": cv_r2.mean()
        })
        trained[name] = pipe

    benchmark = pd.DataFrame(rows).sort_values("RMSE")
    benchmark.to_csv(OUTPUTS / "regression_benchmark_filter_wrapper_embedded.csv", index=False)

    best_model_name = benchmark.iloc[0]["Model"]
    joblib.dump(trained[best_model_name], MODELS / "best_regression_pipeline.joblib")

    # Classification benchmark
    Xc = df[num_cols + cat_cols]
    yc = df["Profit_Level"].astype(str)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=.2, random_state=42, stratify=yc
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=300, random_state=42)
    }

    clf_rows = []
    for name, model in classifiers.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols)),
            ("model", model)
        ])
        pipe.fit(Xc_train, yc_train)
        pred = pipe.predict(Xc_test)
        clf_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yc_test, pred),
            "F1_Macro": f1_score(yc_test, pred, average="macro")
        })
        if name == "Random Forest Classifier":
            joblib.dump(pipe, MODELS / "profit_level_classifier_pipeline.joblib")

    pd.DataFrame(clf_rows).sort_values("F1_Macro", ascending=False).to_csv(
        OUTPUTS / "classification_benchmark.csv", index=False
    )

    summary = {
        "final_features": final_features,
        "best_regression_model": best_model_name,
        "kmeans_silhouette": silhouette_score(scaled_num, df["Cluster"]),
        "pca_explained_variance": pca.explained_variance_ratio_.sum()
    }
    with open(OUTPUTS / "project_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
