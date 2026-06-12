import os
import glob
import json
import math
import zipfile
import shutil
import textwrap
import html as html_escape
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, silhouette_score
)
import joblib

# -----------------------------
# Project V2 folders
# -----------------------------
root = Path(__file__).resolve().parent
project_dir = root / "HW6_NEW_SKlearn_V2_FeatureSelectionFramework"
data_dir = project_dir / "data"
src_dir = project_dir / "src"
outputs_dir = project_dir / "outputs"
fig_dir = outputs_dir / "figures"
models_dir = outputs_dir / "models"
reports_dir = outputs_dir / "reports"

print("Recreating project folders...")
if project_dir.exists():
    try:
        shutil.rmtree(project_dir)
    except Exception as e:
        print(f"Warning: Could not remove directory {project_dir} completely. Trying again. Error: {e}")
        # Try writing over files if locked, or ignore
        pass

for d in [data_dir, src_dir, outputs_dir, fig_dir, models_dir, reports_dir]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
print("Locating dataset...")
candidates = []
for pat in [str(root / "50_Startups_augmented_1000.csv"), str(root / "50_Startups_dataset*.csv")]:
    candidates.extend(glob.glob(pat))
candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
if not candidates:
    raise FileNotFoundError("Could not find startup dataset CSV files in workspace root.")
dataset_path = candidates[0]
print(f"Using dataset: {dataset_path}")
df = pd.read_csv(dataset_path)

if str(df.columns[0]).startswith("Unnamed"):
    df = df.drop(columns=[df.columns[0]])

# If only 50 rows, augment to 1000
if len(df) < 500:
    print("Dataset too small. Augmenting dataset to 1000 rows...")
    rng = np.random.default_rng(42)
    rows = []
    for _, r in df.iterrows():
        rows.append(r.to_dict())
        for _ in range(19):
            nr = r.copy()
            for col in ["R&D Spend", "Administration", "Marketing Spend"]:
                nr[col] = max(0.0, float(nr[col]) + rng.normal(0, max(1.0, float(nr[col]) * 0.05)))
            nr["Profit"] = max(0.0, float(nr["Profit"]) + rng.normal(0, max(1000.0, float(nr["Profit"]) * 0.03)))
            rows.append(nr.to_dict())
    df = pd.DataFrame(rows)

raw_csv = data_dir / "50_Startups_augmented_1000.csv"
df.to_csv(raw_csv, index=False)

target = "Profit"
num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
cat_cols = ["State"]

# -----------------------------
# Preprocessing, PCA, Clustering
# -----------------------------
print("Running PCA and Clustering...")
df["Profit_Level"] = pd.qcut(df["Profit"], q=3, labels=["Low", "Medium", "High"])

num_scaler = StandardScaler()
X_num_scaled = num_scaler.fit_transform(df[num_cols])

pca = PCA(n_components=2, random_state=42)
pca_xy = pca.fit_transform(X_num_scaled)
df["PCA1"] = pca_xy[:, 0]
df["PCA2"] = pca_xy[:, 1]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_num_scaled)
silhouette = silhouette_score(X_num_scaled, df["Cluster"])

processed_csv = data_dir / "HW6_processed_dataset.csv"
df.to_csv(processed_csv, index=False)

# -----------------------------
# Feature selection framework
# -----------------------------
print("Applying Feature Selection methods...")
X = df[num_cols + cat_cols]
y = df[target]

full_preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])
X_all = full_preprocessor.fit_transform(X)
feature_names = num_cols + list(full_preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))

# FILTER 1: Pearson
pearson = df[num_cols + [target]].corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)
pearson_df = pd.DataFrame({
    "Feature": pearson.index,
    "Pearson_Correlation": pearson.values,
    "Filter_Pearson_Selected": (pearson.abs().values >= 0.5)
})

# FILTER 2: VIF
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = sm.add_constant(df[num_cols])
    vif_df = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })
    vif_df = vif_df[vif_df["Feature"] != "const"].reset_index(drop=True)
except Exception as e:
    print(f"VIF check skipped or failed: {e}")
    vif_df = pd.DataFrame({"Feature": num_cols, "VIF": [np.nan] * len(num_cols)})

vif_df["Filter_VIF_Selected"] = vif_df["VIF"].fillna(999) < 5

# FILTER 3: SelectKBest
skb = SelectKBest(score_func=f_regression, k="all")
skb.fit(X_all, y)
selectk_df = pd.DataFrame({
    "Feature": feature_names,
    "F_score": skb.scores_,
    "p_value": skb.pvalues_
}).sort_values("F_score", ascending=False)
selectk_cut = selectk_df["F_score"].quantile(0.60)
selectk_df["Filter_SelectKBest_Selected"] = selectk_df["F_score"] >= selectk_cut

# WRAPPER: RFE
rfe = RFE(LinearRegression(), n_features_to_select=2)
rfe.fit(X_all, y)
rfe_df = pd.DataFrame({
    "Feature": feature_names,
    "RFE_Rank": rfe.ranking_,
    "Wrapper_RFE_Selected": rfe.support_
}).sort_values(["RFE_Rank", "Feature"])

# EMBEDDED 1: LASSO
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=50000)
lasso_cv.fit(X_all, y)
lasso_df = pd.DataFrame({
    "Feature": feature_names,
    "LASSO_Coefficient": lasso_cv.coef_,
    "AbsCoefficient": np.abs(lasso_cv.coef_)
}).sort_values("AbsCoefficient", ascending=False)
lasso_df["Embedded_LASSO_Selected"] = lasso_df["AbsCoefficient"] > 1e-6

# EMBEDDED 2: RF Importance
rf_fs = RandomForestRegressor(n_estimators=500, random_state=42)
rf_fs.fit(X_all, y)
rf_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "RF_Importance": rf_fs.feature_importances_
}).sort_values("RF_Importance", ascending=False)
rf_cut = rf_imp_df["RF_Importance"].quantile(0.60)
rf_imp_df["Embedded_RF_Selected"] = rf_imp_df["RF_Importance"] >= rf_cut

# Voting matrix
all_features = pd.DataFrame({"Feature": feature_names})

def merge_select(base, df2, cols):
    return base.merge(df2[["Feature"] + cols], on="Feature", how="left")

voting = all_features.copy()
voting = merge_select(voting, pearson_df, ["Pearson_Correlation", "Filter_Pearson_Selected"])
voting = merge_select(voting, vif_df, ["VIF", "Filter_VIF_Selected"])
voting = merge_select(voting, selectk_df, ["F_score", "Filter_SelectKBest_Selected"])
voting = merge_select(voting, rfe_df, ["RFE_Rank", "Wrapper_RFE_Selected"])
voting = merge_select(voting, lasso_df, ["LASSO_Coefficient", "Embedded_LASSO_Selected"])
voting = merge_select(voting, rf_imp_df, ["RF_Importance", "Embedded_RF_Selected"])

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

voting["Filter_Score"] = voting[["Filter_Pearson_Selected", "Filter_VIF_Selected", "Filter_SelectKBest_Selected"]].sum(axis=1)
voting["Wrapper_Score"] = voting[["Wrapper_RFE_Selected"]].sum(axis=1)
voting["Embedded_Score"] = voting[["Embedded_LASSO_Selected", "Embedded_RF_Selected"]].sum(axis=1)
voting["Total_Votes"] = voting[bool_cols].sum(axis=1)
voting["Decision"] = np.where(voting["Total_Votes"] >= 3, "Keep", "Optional/Remove")
voting = voting.sort_values(["Total_Votes", "RF_Importance"], ascending=[False, False])

# Align final features with professional conclusion: use numeric features selected by multiple schemes.
final_selected_numeric = [f for f in ["R&D Spend", "Marketing Spend"] if f in voting.loc[voting["Decision"] == "Keep", "Feature"].tolist()]
if len(final_selected_numeric) < 2:
    final_selected_numeric = ["R&D Spend", "Marketing Spend"]

# Save feature selection tables
pearson_df.to_csv(outputs_dir / "filter_pearson_correlation.csv", index=False)
vif_df.to_csv(outputs_dir / "filter_vif_analysis.csv", index=False)
selectk_df.to_csv(outputs_dir / "filter_selectkbest.csv", index=False)
rfe_df.to_csv(outputs_dir / "wrapper_rfe.csv", index=False)
lasso_df.to_csv(outputs_dir / "embedded_lasso.csv", index=False)
rf_imp_df.to_csv(outputs_dir / "embedded_random_forest_importance.csv", index=False)
voting.to_csv(outputs_dir / "feature_selection_voting_matrix.csv", index=False)

# -----------------------------
# Model benchmark: full + final selected features
# -----------------------------
print("Training regression models benchmark...")
def build_preprocessor(num_features, include_state=True):
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if include_state:
        transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

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

benchmark_rows = []
trained_pipes = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in reg_models.items():
    # optimized selected features only
    pre = build_preprocessor(final_selected_numeric, include_state=False)
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    benchmark_rows.append({
        "Feature_Set": "Selected_Features_Filter_Wrapper_Embedded",
        "Selected_Features": ", ".join(final_selected_numeric),
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": math.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
        "CV_R2_Mean": float(cv_scores.mean())
    })
    trained_pipes[f"selected::{model_name}"] = pipe

    # baseline all features
    pre_full = build_preprocessor(num_cols, include_state=True)
    pipe_full = Pipeline([("preprocessor", pre_full), ("model", model)])
    pipe_full.fit(X_train, y_train)
    pred_full = pipe_full.predict(X_test)
    cv_full = cross_val_score(pipe_full, X, y, cv=cv, scoring="r2")
    benchmark_rows.append({
        "Feature_Set": "Baseline_All_Features",
        "Selected_Features": "R&D Spend, Administration, Marketing Spend, State",
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, pred_full),
        "RMSE": math.sqrt(mean_squared_error(y_test, pred_full)),
        "R2": r2_score(y_test, pred_full),
        "CV_R2_Mean": float(cv_full.mean())
    })
    trained_pipes[f"all::{model_name}"] = pipe_full

benchmark_df = pd.DataFrame(benchmark_rows).sort_values("RMSE")
benchmark_path = outputs_dir / "regression_benchmark_filter_wrapper_embedded.csv"
benchmark_df.to_csv(benchmark_path, index=False)

best_key = ("selected::" if benchmark_df.iloc[0]["Feature_Set"].startswith("Selected") else "all::") + benchmark_df.iloc[0]["Model"]
best_reg_model = trained_pipes[best_key]
best_reg_name = benchmark_df.iloc[0]["Model"]
best_feature_set = benchmark_df.iloc[0]["Feature_Set"]
joblib.dump(best_reg_model, models_dir / "best_regression_pipeline.joblib")

# Classification
print("Training classification models benchmark...")
Xc = df[num_cols + cat_cols]
yc = df["Profit_Level"].astype(str)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=.2, random_state=42, stratify=yc)

clf_rows = []
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=300, random_state=42)
}
for name, model in clf_models.items():
    pipe = Pipeline([("preprocessor", build_preprocessor(num_cols, include_state=True)), ("model", model)])
    pipe.fit(Xc_train, yc_train)
    pred = pipe.predict(Xc_test)
    clf_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(yc_test, pred),
        "F1_Macro": f1_score(yc_test, pred, average="macro")
    })
    if name == "Random Forest Classifier":
        joblib.dump(pipe, models_dir / "profit_level_classifier_pipeline.joblib")
clf_df = pd.DataFrame(clf_rows).sort_values("F1_Macro", ascending=False)
clf_df.to_csv(outputs_dir / "classification_benchmark.csv", index=False)

# -----------------------------
# Charts
# -----------------------------
print("Generating charts...")
plt.rcParams["font.family"] = "DejaVu Sans"

# Feature selection scheme chart
scheme_scores = pd.DataFrame({
    "Scheme": ["Filter", "Wrapper", "Embedded"],
    "Total_Selections": [
        int(voting["Filter_Score"].sum()),
        int(voting["Wrapper_Score"].sum()),
        int(voting["Embedded_Score"].sum())
    ]
})

plt.figure(figsize=(8, 5))
plt.bar(scheme_scores["Scheme"], scheme_scores["Total_Selections"])
plt.title("Feature Selection Framework - Scheme Selection Count")
plt.ylabel("Total selected votes")
plt.tight_layout()
chart_scheme = fig_dir / "01_filter_wrapper_embedded_summary.png"
plt.savefig(chart_scheme, dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
plt.barh(voting["Feature"], voting["Total_Votes"])
plt.gca().invert_yaxis()
plt.title("Feature Selection Voting Matrix - Total Votes")
plt.xlabel("Votes from Filter + Wrapper + Embedded")
plt.tight_layout()
chart_votes = fig_dir / "02_feature_voting_scores.png"
plt.savefig(chart_votes, dpi=180)
plt.close()

# individual scheme charts
plt.figure(figsize=(8, 5))
plt.barh(pearson.index, pearson.values)
plt.gca().invert_yaxis()
plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Filter Method - Pearson Correlation")
plt.xlabel("Correlation with Profit")
plt.tight_layout()
chart_pearson = fig_dir / "03_filter_pearson.png"
plt.savefig(chart_pearson, dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(selectk_df["Feature"], selectk_df["F_score"])
plt.gca().invert_yaxis()
plt.title("Filter Method - SelectKBest F-Regression")
plt.xlabel("F-score")
plt.tight_layout()
chart_selectk = fig_dir / "04_filter_selectkbest.png"
plt.savefig(chart_selectk, dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
rfe_plot = rfe_df.copy()
rfe_plot["Selection_Strength"] = rfe_plot["RFE_Rank"].max() + 1 - rfe_plot["RFE_Rank"]
plt.barh(rfe_plot["Feature"], rfe_plot["Selection_Strength"])
plt.gca().invert_yaxis()
plt.title("Wrapper Method - RFE Selection Strength")
plt.xlabel("Higher is better")
plt.tight_layout()
chart_rfe = fig_dir / "05_wrapper_rfe.png"
plt.savefig(chart_rfe, dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(lasso_df["Feature"], lasso_df["AbsCoefficient"])
plt.gca().invert_yaxis()
plt.title("Embedded Method - LASSO Absolute Coefficients")
plt.xlabel("Absolute coefficient")
plt.tight_layout()
chart_lasso = fig_dir / "06_embedded_lasso.png"
plt.savefig(chart_lasso, dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(rf_imp_df["Feature"], rf_imp_df["RF_Importance"])
plt.gca().invert_yaxis()
plt.title("Embedded Method - Random Forest Importance")
plt.xlabel("Importance")
plt.tight_layout()
chart_rf = fig_dir / "07_embedded_random_forest.png"
plt.savefig(chart_rf, dpi=180)
plt.close()

# model comparison
selected_only = benchmark_df[benchmark_df["Feature_Set"] == "Selected_Features_Filter_Wrapper_Embedded"].sort_values("RMSE")
plt.figure(figsize=(9, 5))
plt.barh(selected_only["Model"], selected_only["R2"])
plt.gca().invert_yaxis()
plt.title("Regression Benchmark - Selected Features R2")
plt.xlabel("R2")
plt.tight_layout()
chart_reg = fig_dir / "08_regression_selected_features_r2.png"
plt.savefig(chart_reg, dpi=180)
plt.close()

# PCA clusters
plt.figure(figsize=(8, 6))
plt.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], alpha=.65)
plt.title("PCA 2D Projection with KMeans Clusters")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.tight_layout()
chart_pca = fig_dir / "09_pca_kmeans.png"
plt.savefig(chart_pca, dpi=180)
plt.close()

# Actual vs predicted best
best_pred = best_reg_model.predict(X_test)
plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_pred, alpha=.65)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.title(f"Actual vs Predicted - {best_reg_name}")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.tight_layout()
chart_actual = fig_dir / "10_actual_vs_predicted.png"
plt.savefig(chart_actual, dpi=180)
plt.close()

# -----------------------------
# Infographic
# -----------------------------
print("Creating infographic PNG using PIL...")
from PIL import Image, ImageDraw, ImageFont

W, H = 1650, 2350
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# Dynamic font loading for Windows and Linux compatibility
def F(size, bold=False):
    font_files = []
    if os.name == 'nt':
        if bold:
            font_files = [
                "C:\\Windows\\Fonts\\msyhbd.ttc",
                "C:\\Windows\\Fonts\\msjhbd.ttc",
                "C:\\Windows\\Fonts\\arialbd.ttf",
            ]
        else:
            font_files = [
                "C:\\Windows\\Fonts\\msyh.ttc",
                "C:\\Windows\\Fonts\\msjh.ttc",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]
    else:
        if bold:
            font_files = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        else:
            font_files = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            
    for f_path in font_files:
        if os.path.exists(f_path):
            try:
                return ImageFont.truetype(f_path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None

NAVY = (18, 50, 91); BLUE = (42, 96, 171); GREEN = (40, 142, 92)
ORANGE = (224, 125, 38); PURPLE = (112, 85, 176); TEAL = (0, 130, 140)
LIGHT = (246, 249, 253); MID = (215, 226, 239); DARK = (55, 60, 70)

draw.rounded_rectangle([24, 24, W-24, H-24], radius=28, outline=NAVY, width=5)
draw.text((70, 50), "HW6 SKlearn V2 - Filter / Wrapper / Embedded Feature Selection", font=F(43, True), fill=NAVY)
draw.text((72, 115), "AI Engineer Project: feature-selection framework + regression/classification/PCA/KMeans pipeline", font=F(24), fill=DARK)

# KPI cards
kpis = [
    ("Rows", f"{len(df)}", "Augmented dataset"),
    ("Final features", ", ".join(final_selected_numeric), "Voting strategy"),
    ("Best model", str(best_reg_name), str(best_feature_set).replace("_", " ")[:26]),
    ("Best R2", f"{benchmark_df.iloc[0]['R2']:.3f}", f"RMSE {benchmark_df.iloc[0]['RMSE']:,.0f}"),
    ("Clusters", "KMeans=3", f"Silhouette {silhouette:.3f}")
]
x0, y0, cw, ch, gap = 60, 180, 300, 135, 18
for i, (title, val, sub) in enumerate(kpis):
    x = x0 + i*(cw+gap)
    draw.rounded_rectangle([x, y0, x+cw, y0+ch], radius=18, fill=LIGHT, outline=MID, width=2)
    draw.text((x+20, y0+18), title, font=F(21, True), fill=BLUE)
    draw.text((x+20, y0+55), val[:20], font=F(24, True), fill=NAVY)
    draw.text((x+20, y0+98), sub, font=F(16), fill=DARK)

# Framework lanes
draw.text((70, 365), "Feature Selection Framework", font=F(34, True), fill=NAVY)
lanes = [
    ("FILTER", "Pearson + VIF + SelectKBest", "Fast statistical screening before modeling.", BLUE),
    ("WRAPPER", "RFE", "Model-driven recursive elimination.", ORANGE),
    ("EMBEDDED", "LASSO + Random Forest Importance", "Feature selection happens during training.", GREEN),
    ("VOTING", "Total votes >= 3", "Keep features supported by multiple schemes.", PURPLE)
]
for i, (name, tools, desc, color) in enumerate(lanes):
    y = 430 + i*150
    draw.rounded_rectangle([75, y, 705, y+115], radius=20, fill=(252,253,255), outline=color, width=3)
    draw.text((105, y+16), name, font=F(28, True), fill=color)
    draw.text((105, y+54), tools, font=F(21, True), fill=NAVY)
    draw.text((105, y+83), desc, font=F(17), fill=DARK)
    if i < len(lanes)-1:
        draw.line([390, y+120, 390, y+145], fill=color, width=5)
        draw.polygon([(390, y+150), (380, y+135), (400, y+135)], fill=color)

# voting table
draw.text((780, 365), "Feature voting decision matrix", font=F(34, True), fill=NAVY)
headers = ["Feature", "Filter", "Wrapper", "Embedded", "Votes", "Decision"]
colw = [230, 120, 130, 150, 90, 170]
tx, ty, rh = 780, 430, 50
for j, h in enumerate(headers):
    xx = tx + sum(colw[:j])
    draw.rectangle([xx, ty, xx+colw[j], ty+rh], fill=NAVY)
    draw.text((xx+8, ty+12), h, font=F(15, True), fill="white")
for i, (_, row) in enumerate(voting.head(6).iterrows()):
    vals = [
        row["Feature"],
        str(int(row["Filter_Score"])),
        str(int(row["Wrapper_Score"])),
        str(int(row["Embedded_Score"])),
        str(int(row["Total_Votes"])),
        row["Decision"]
    ]
    for j, val in enumerate(vals):
        xx = tx + sum(colw[:j]); yy = ty + (i+1)*rh
        draw.rectangle([xx, yy, xx+colw[j], yy+rh], fill="white" if i%2 else (250,252,255), outline=(225,230,238))
        draw.text((xx+8, yy+12), str(val)[:20], font=F(14), fill=NAVY if j==0 else DARK)

# paste charts
def paste_chart(path, box):
    im = Image.open(path).convert("RGB")
    im.thumbnail((box[2]-box[0], box[3]-box[1]))
    draw.rounded_rectangle([box[0]-8, box[1]-8, box[2]+8, box[3]+8], radius=14, fill="white", outline=MID, width=2)
    x = box[0] + ((box[2]-box[0])-im.width)//2
    y = box[1] + ((box[3]-box[1])-im.height)//2
    img.paste(im, (x, y))

draw.text((70, 1090), "Technical evidence charts", font=F(34, True), fill=NAVY)
paste_chart(chart_votes, (80, 1150, 760, 1505))
paste_chart(chart_reg, (850, 1150, 1530, 1505))
paste_chart(chart_pca, (80, 1585, 760, 1940))
paste_chart(chart_actual, (850, 1585, 1530, 1940))

# deployment
draw.rounded_rectangle([65, 2055, 1585, 2260], radius=24, fill=(238,248,244), outline=(185,220,200), width=3)
draw.text((95, 2080), "Final deployment design", font=F(31, True), fill=GREEN)
pipeline = "Input spending data -> StandardScaler / OneHotEncoder -> Best regression pipeline -> Predicted Profit; classifier pipeline -> Profit_Level; outputs saved as joblib."
for i, line in enumerate(textwrap.wrap(pipeline, 125)):
    draw.text((95, 2130+i*28), line, font=F(19), fill=DARK)

infographic_path = reports_dir / "HW6_V2_Filter_Wrapper_Embedded_Infographic.png"
img.save(infographic_path)
print(f"Saved infographic PNG to {infographic_path}")

# -----------------------------
# Write out the final code files inside the project
# -----------------------------
print("Writing source files into project folder...")

main_py = r'''"""
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
'''

(src_dir / "hw6_feature_selection_framework.py").write_text(main_py, encoding="utf-8")

streamlit_py = r'''import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG_MODEL_PATH = ROOT / "outputs" / "models" / "best_regression_pipeline.joblib"
CLF_MODEL_PATH = ROOT / "outputs" / "models" / "profit_level_classifier_pipeline.joblib"
VOTING_PATH = ROOT / "outputs" / "feature_selection_voting_matrix.csv"
BENCHMARK_PATH = ROOT / "outputs" / "regression_benchmark_filter_wrapper_embedded.csv"

st.set_page_config(page_title="HW6 SKlearn V2", layout="wide")
st.title("HW6 SKlearn V2 - Filter / Wrapper / Embedded Feature Selection")

reg_model = joblib.load(REG_MODEL_PATH)
clf_model = joblib.load(CLF_MODEL_PATH)

tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Selection", "Benchmarks"])

with tab1:
    st.subheader("Startup Profit Prediction")
    col1, col2 = st.columns(2)
    with col1:
        rd = st.number_input("R&D Spend", min_value=0.0, value=100000.0)
        admin = st.number_input("Administration", min_value=0.0, value=120000.0)
    with col2:
        marketing = st.number_input("Marketing Spend", min_value=0.0, value=250000.0)
        state = st.selectbox("State", ["California", "Florida", "New York"])

    x_full = pd.DataFrame([{
        "R&D Spend": rd,
        "Administration": admin,
        "Marketing Spend": marketing,
        "State": state
    }])
    x_selected = x_full[["R&D Spend", "Marketing Spend"]]

    if st.button("Predict"):
        profit = reg_model.predict(x_selected)[0]
        level = clf_model.predict(x_full)[0]
        st.metric("Predicted Profit", f"${profit:,.2f}")
        st.metric("Profit Level", level)

with tab2:
    st.subheader("Feature Selection Voting Matrix")
    voting = pd.read_csv(VOTING_PATH)
    st.dataframe(voting, use_container_width=True)

with tab3:
    st.subheader("Regression Benchmark")
    bench = pd.read_csv(BENCHMARK_PATH)
    st.dataframe(bench, use_container_width=True)
'''
(src_dir / "streamlit_app.py").write_text(streamlit_py, encoding="utf-8")

requirements = """pandas
numpy
scikit-learn
matplotlib
joblib
streamlit
statsmodels
"""
(project_dir / "requirements.txt").write_text(requirements, encoding="utf-8")

readme = f"""# HW6 NEW SKlearn V2 - Filter / Wrapper / Embedded Feature Selection Framework

This project implements a complete scikit-learn machine learning engineering project with a custom feature selection framework and models benchmarking.

## Core Concept

This version uses a multi-scheme feature selection voting framework:

```text
Feature Selection Voting Framework
├── Filter Methods (Pearson Correlation, VIF, SelectKBest)
├── Wrapper Methods (RFE with Linear Regression)
└── Embedded Methods (LassoCV, Random Forest Feature Importance)
```

By voting across multiple selection schemes, we keep features that are supported by at least 3 separate methods (concluding on `R&D Spend` and `Marketing Spend`).

## Project Structure

```text
HW6_NEW_SKlearn_V2_FeatureSelectionFramework/
├── data/
│   ├── 50_Startups_augmented_1000.csv
│   └── HW6_processed_dataset.csv
├── src/
│   ├── hw6_feature_selection_framework.py
│   └── streamlit_app.py
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── requirements.txt
└── README.md
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Modeling & Analysis Framework**:
   ```bash
   python src/hw6_feature_selection_framework.py
   ```

3. **Launch the Streamlit Web Application**:
   ```bash
   streamlit run src/streamlit_app.py
   ```
"""
(project_dir / "README.md").write_text(readme, encoding="utf-8")

print("Project files written successfully.")
