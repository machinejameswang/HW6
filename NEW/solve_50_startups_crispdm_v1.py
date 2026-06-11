"""
Kaggle 50 Startups CRISP-DM Scikit-learn Project

This script follows the six CRISP-DM steps:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

Dataset file expected in the same folder:
    50_Startups_dataset.csv
"""

from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ModuleNotFoundError as import_error:
    DEPENDENCY_IMPORT_ERROR = import_error
else:
    DEPENDENCY_IMPORT_ERROR = None


DATA_FILE = "50_Startups_dataset.csv"
TARGET_COLUMN = "Profit"
MODEL_OUTPUT_FILE = "startup_profit_model_v1.pkl"

NUMERICAL_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
REQUIRED_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]


def print_section(title):
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def business_understanding():
    """CRISP-DM Step 1: Explain the business problem."""
    print_section("CRISP-DM Step 1: Business Understanding")
    print(
        """
Business problem:
The goal is to predict startup Profit using spending information from the
Kaggle 50 Startups dataset. The available predictors are R&D Spend,
Administration, Marketing Spend, and State.

Why this matters:
Startup resources are usually limited, so founders, investors, and analysts
need to understand which spending patterns are associated with higher expected
profit. A predictive model can support budgeting, investment review, and
scenario planning.

Machine learning framing:
This is a supervised learning regression problem because the target variable,
Profit, is a continuous numeric value.

Important expert note:
This dataset has only 50 rows, so results should be interpreted as predictive
associations, not causal proof. The model can suggest patterns, but it should
not be used to claim that one spending category directly causes profit changes.
""".strip()
    )


def load_dataset(file_path):
    """Load dataset and handle missing file errors."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Place {DATA_FILE} in the same folder "
            "as this script, then run again."
        )
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def check_required_columns(df):
    """Validate that all required columns are available before modeling."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
            + f"\nRequired columns are: {', '.join(REQUIRED_COLUMNS)}"
        )


def data_understanding(df):
    """CRISP-DM Step 2: Print dataset overview and exploratory checks."""
    print_section("CRISP-DM Step 2: Data Understanding")

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nData types and non-null counts:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nState distribution:")
    print(df["State"].value_counts())

    print("\nCorrelation matrix for numerical columns:")
    print(df.corr(numeric_only=True))

    print("\nProfit by State:")
    print(df.groupby("State")["Profit"].agg(["count", "mean", "min", "max", "std"]))


def feature_analysis():
    """Print expert-level feature analysis without causal overclaiming."""
    print_section("Expert Feature Analysis")
    print(
        """
R&D Spend:
- Role: Core innovation factor
- Expected importance: High
- Recommendation: Always keep
- Interpretation: R&D Spend is expected to be the strongest predictor because
  it reflects product development, innovation ability, and technical
  competitiveness.

Marketing Spend:
- Role: Market expansion factor
- Expected importance: Medium to high
- Recommendation: Keep, but check correlation with R&D Spend
- Interpretation: Marketing Spend may add predictive value, but it may also
  overlap with company size and R&D Spend. Avoid interpreting it as an
  independent causal factor.

Administration:
- Role: Operating cost and company scale factor
- Expected importance: Low to medium
- Recommendation: Keep first, evaluate later
- Interpretation: Administration may be weaker because it does not directly
  create revenue, but it can still reflect company scale and operational
  maturity.

State:
- Role: Regional auxiliary factor
- Expected importance: Low to medium
- Recommendation: Use One-Hot Encoding and avoid overinterpretation
- Interpretation: State may reflect regional business environment, but the
  dataset is small, so it should be treated as an auxiliary variable only.
""".strip()
    )


def build_pipeline(feature_columns):
    """Build sklearn preprocessing and regression pipeline."""
    selected_numerical = [
        column for column in feature_columns if column in NUMERICAL_FEATURES
    ]
    selected_categorical = [
        column for column in feature_columns if column in CATEGORICAL_FEATURES
    ]

    transformers = []
    if selected_numerical:
        transformers.append(("num", "passthrough", selected_numerical))
    if selected_categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                selected_categorical,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def evaluate_train_test(pipeline, X_train, X_test, y_train, y_test):
    """Evaluate model with R2, MAE, and RMSE on the test set."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "Test R2": r2_score(y_test, y_pred),
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def evaluate_cross_validation(pipeline, X, y):
    """Evaluate model with 5-fold CV using R2 and RMSE."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="r2",
    )
    cv_negative_rmse_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    cv_rmse_scores = -cv_negative_rmse_scores

    return {
        "CV R2 Mean": cv_r2_scores.mean(),
        "CV R2 Std": cv_r2_scores.std(),
        "CV RMSE Mean": cv_rmse_scores.mean(),
        "CV RMSE Std": cv_rmse_scores.std(),
    }


def run_model_experiments(df):
    """CRISP-DM Steps 3-5: Prepare data, train models, and evaluate them."""
    print_section("CRISP-DM Step 3: Data Preparation")

    y = df[TARGET_COLUMN]

    model_experiments = {
        "R&D Only": {
            "features": ["R&D Spend"],
            "purpose": "Check the predictive power of the core feature.",
        },
        "R&D + Marketing": {
            "features": ["R&D Spend", "Marketing Spend"],
            "purpose": "Check whether Marketing adds value beyond R&D.",
        },
        "Numerical Features": {
            "features": ["R&D Spend", "Marketing Spend", "Administration"],
            "purpose": "Check whether Administration improves prediction.",
        },
        "All Features": {
            "features": [
                "R&D Spend",
                "Marketing Spend",
                "Administration",
                "State",
            ],
            "purpose": "Check whether State improves prediction.",
        },
    }

    print("Target variable: Profit")
    print("Train-test split: test_size=0.2, random_state=42")
    print("Categorical encoding for State: OneHotEncoder(drop='first')")

    print_section("CRISP-DM Step 4: Modeling")
    print("Primary model: LinearRegression from scikit-learn")

    print_section("CRISP-DM Step 5: Evaluation")

    results = []
    fitted_pipelines = {}

    for model_name, experiment in model_experiments.items():
        feature_columns = experiment["features"]
        X = df[feature_columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        pipeline = build_pipeline(feature_columns)
        train_test_metrics = evaluate_train_test(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        cv_metrics = evaluate_cross_validation(
            build_pipeline(feature_columns),
            X,
            y,
        )

        fitted_pipelines[model_name] = pipeline

        results.append(
            {
                "Model": model_name,
                "Features": ", ".join(feature_columns),
                "Purpose": experiment["purpose"],
                **train_test_metrics,
                **cv_metrics,
            }
        )

    results_df = pd.DataFrame(results)
    metric_columns = [
        "Test R2",
        "Test MAE",
        "Test RMSE",
        "CV R2 Mean",
        "CV R2 Std",
        "CV RMSE Mean",
        "CV RMSE Std",
    ]
    results_df[metric_columns] = results_df[metric_columns].round(4)

    print("\nModel comparison table:")
    print(results_df.to_string(index=False))

    return results_df, fitted_pipelines, model_experiments


def select_final_model(results_df, model_experiments):
    """Select best model based on CV R2 Mean."""
    best_row = results_df.sort_values(
        by=["CV R2 Mean", "CV RMSE Mean"],
        ascending=[False, True],
    ).iloc[0]
    best_model_name = best_row["Model"]
    best_features = model_experiments[best_model_name]["features"]

    print_section("Final Model Selection")
    print(f"Selected final model: {best_model_name}")
    print(f"Selected features: {best_features}")
    print(f"Selection rule: highest CV R2 Mean, with simpler models preferred if close.")

    all_features_row = results_df[results_df["Model"] == "All Features"].iloc[0]
    numerical_row = results_df[results_df["Model"] == "Numerical Features"].iloc[0]
    state_gain = all_features_row["CV R2 Mean"] - numerical_row["CV R2 Mean"]

    rd_only_row = results_df[results_df["Model"] == "R&D Only"].iloc[0]
    best_gain_over_rd = best_row["CV R2 Mean"] - rd_only_row["CV R2 Mean"]

    print("\nInterpretation:")
    print(
        f"- State CV R2 gain over numerical-only model: {state_gain:.4f}. "
        "If this value is small or negative, State does not add meaningful "
        "predictive value in this dataset."
    )
    print(
        f"- Final model CV R2 gain over R&D-only model: {best_gain_over_rd:.4f}. "
        "If this value is small, R&D Spend dominates the prediction."
    )
    print(
        """
- The model comparison shows that R&D Spend is expected to be the most important
  predictor of Profit. Marketing Spend may provide additional predictive value,
  while Administration may be weaker but still useful as a scale-related factor.
  State is treated as an auxiliary categorical feature and encoded using
  One-Hot Encoding. Because the dataset contains only 50 observations, the
  results should be interpreted as predictive associations rather than causal
  conclusions.
""".strip()
    )

    return best_model_name, best_features


def deployment_simulation(final_pipeline):
    """CRISP-DM Step 6: Predict Profit for a new startup."""
    print_section("CRISP-DM Step 6: Deployment Simulation")

    sample_input = pd.DataFrame(
        [
            {
                "R&D Spend": 120000,
                "Administration": 130000,
                "Marketing Spend": 250000,
                "State": "New York",
            }
        ]
    )

    prediction = final_pipeline.predict(sample_input)[0]

    print("\nNew startup input:")
    print(sample_input.to_string(index=False))
    print(f"\nPredicted Profit: {prediction:,.2f}")

    return prediction


def save_model(final_pipeline, output_file=MODEL_OUTPUT_FILE):
    """Save final pipeline using joblib."""
    joblib.dump(final_pipeline, output_file)
    print(f"\nSaved final model to: {output_file}")


def main():
    """Run the complete CRISP-DM workflow."""
    if DEPENDENCY_IMPORT_ERROR is not None:
        print_section("Dependency Error")
        print(f"Missing Python package: {DEPENDENCY_IMPORT_ERROR.name}")
        print("Install the required packages, then run this script again:")
        print("pip install pandas numpy joblib scikit-learn")
        return

    business_understanding()

    try:
        df = load_dataset(DATA_FILE)
        check_required_columns(df)
    except (FileNotFoundError, ValueError) as error:
        print_section("Dataset Error")
        print(error)
        return

    data_understanding(df)
    feature_analysis()

    results_df, fitted_pipelines, model_experiments = run_model_experiments(df)
    best_model_name, best_features = select_final_model(results_df, model_experiments)

    final_pipeline = build_pipeline(best_features)
    final_pipeline.fit(df[best_features], df[TARGET_COLUMN])

    deployment_simulation(final_pipeline)
    save_model(final_pipeline)

    print_section("Workflow Complete")
    print("The full CRISP-DM workflow has finished successfully.")


if __name__ == "__main__":
    main()
