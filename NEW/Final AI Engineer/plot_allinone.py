import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set matplotlib style for professional appearance
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# Load dataset
df = pd.read_csv("50_Startups_dataset.csv")
if str(df.columns[0]).startswith("Unnamed"):
    df = df.drop(columns=[df.columns[0]])

target = "Profit"
num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
cat_cols = ["State"]

X = df[num_cols + cat_cols]
y = df[target]

# Split data using the exact split matching the screenshot: test_size=0.2, random_state=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

full_preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])
X_train_scaled = full_preprocessor.fit_transform(X_train)
X_test_scaled = full_preprocessor.transform(X_test)

feature_names = num_cols + list(full_preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
feature_names = [f.replace("State_", "") for f in feature_names]
feat_to_idx = {name: i for i, name in enumerate(feature_names)}

# Define the exact rankings from the user's screenshot
rankings = {
    "SFS (Forward)": ["R&D Spend", "Marketing Spend", "New York", "Florida", "Administration"],
    "RFE": ["R&D Spend", "Marketing Spend", "Administration", "Florida", "New York"],
    "SelectKBest": ["R&D Spend", "Marketing Spend", "Administration", "New York", "Florida"],
    "Lasso (L1)": ["R&D Spend", "Marketing Spend", "Administration", "New York", "Florida"],
    "Random Forest": ["R&D Spend", "Marketing Spend", "Administration", "Florida", "New York"]
}

# Line colors matching standard matplotlib colors
colors = {
    "SFS (Forward)": "C0",
    "RFE": "C1",
    "SelectKBest": "C2",
    "Lasso (L1)": "C3",
    "Random Forest": "C4"
}

# Compute performance curves
x_vals = [1, 2, 3, 4, 5]
results = {}

for name, ranking in rankings.items():
    mse_curve = []
    r2_curve = []
    for k in x_vals:
        subset = ranking[:k]
        idxs = [feat_to_idx[f] for f in subset]
        model = LinearRegression().fit(X_train_scaled[:, idxs], y_train)
        pred = model.predict(X_test_scaled[:, idxs])
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mse_curve.append(mse)
        r2_curve.append(r2)
    results[name] = {"MSE": mse_curve, "R2": r2_curve}

# Create a figure with GridSpec: 2 rows, 2 columns
# Row 0: charts (RMSE and R-squared)
# Row 1: table at the bottom
fig = plt.figure(figsize=(14, 10), dpi=180)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 0.8], hspace=0.35)

ax_mse = fig.add_subplot(gs[0, 0])
ax_r2 = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])

# Plot MSE
for name, curves in results.items():
    ax_mse.plot(x_vals, curves["MSE"], marker="o", color=colors[name], label=name, linewidth=1.5)
ax_mse.set_title("MSE by Number of Features (All Algorithms)", fontweight="bold", pad=10)
ax_mse.set_xlabel("Number of Features", labelpad=8)
ax_mse.set_ylabel("MSE", labelpad=8)
ax_mse.set_xticks(x_vals)
ax_mse.grid(True, linestyle="--", alpha=0.5)
ax_mse.legend(loc="upper right", framealpha=0.8)

# Plot R-squared
for name, curves in results.items():
    ax_r2.plot(x_vals, curves["R2"], marker="o", color=colors[name], label=name, linewidth=1.5)
ax_r2.set_title("R-squared by Number of Features (All Algorithms)", fontweight="bold", pad=10)
ax_r2.set_xlabel("Number of Features", labelpad=8)
ax_r2.set_ylabel("R-squared", labelpad=8)
ax_r2.set_xticks(x_vals)
ax_r2.grid(True, linestyle="--", alpha=0.5)
ax_r2.legend(loc="lower right", framealpha=0.8)

# Set grid limits to match screenshot aesthetics if needed (auto-scale is usually perfect, but we can set ticks)
# ax_mse.set_ylim(...) # removed because MSE scale differs significantly from RMSE
ax_r2.set_ylim(0.935, 0.949)

# Draw Table at the bottom
ax_table.axis("off")

# Prepare cell text and cell colors
cell_text = []
row_colors = []
for name, ranking in rankings.items():
    row = [name] + ranking
    cell_text.append(row)

# Create table
table = ax_table.table(
    cellText=cell_text,
    colLabels=["Algorithm", "Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"],
    loc="center",
    cellLoc="center"
)

# Style table elements
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Apply text coloring for first column to match lines
for i, name in enumerate(rankings.keys()):
    cell = table[(i + 1, 0)]
    cell.get_text().set_color(colors[name])
    cell.get_text().set_weight("bold")

# Apply background color and header formatting
for col_idx in range(6):
    cell = table[(0, col_idx)]
    cell.set_facecolor("#f2f2f2")
    cell.get_text().set_weight("bold")

# Save the final image
output_path = "feature_selection_performance_allione.png"
plt.savefig(output_path, bbox_inches="tight", dpi=180)
plt.close()

print(f"Successfully generated {output_path}!")
