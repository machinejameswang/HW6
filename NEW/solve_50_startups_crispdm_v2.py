"""
Kaggle 50 Startups CRISP-DM Scikit-learn Project v2

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
MODEL_OUTPUT_FILE = "startup_profit_model_v2.pkl"

NUMERICAL_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
REQUIRED_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]


def print_section(title):
    """Print a readable section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def business_understanding():
    """CRISP-DM Step 1: Explain the business problem and expert consensus."""
    print_section("CRISP-DM Step 1: Business Understanding")
    print(
        """
Business problem:
The goal is to predict startup Profit using the Kaggle 50 Startups dataset.
The available features are R&D Spend, Administration, Marketing Spend, and
State. This is a supervised learning regression problem because Profit is a
continuous numeric target.

Why predicting Profit is useful:
Startup resources are limited. Founders, investors, analysts, and managers need
better ways to compare spending plans, evaluate resource allocation, and create
profit scenarios before making decisions. A predictive model can support those
decisions by estimating expected Profit from a given spending profile.

Multidisciplinary expert consensus:
- R&D expert: R&D Spend should be treated as the core predictive feature because
  it reflects product innovation, technical capability, product development, and
  long-term competitiveness.
- Marketing expert: Marketing Spend may help predict Profit because it supports
  brand exposure, customer acquisition, and market expansion, but it should not
  be interpreted as an independent guarantee of higher Profit.
- Sales expert: R&D and Marketing create business value only when they connect
  to customer conversion, revenue generation, and market demand.
- Governor / Regional Policy Expert: State may capture regional business
  environment, labor cost, tax policy, talent density, startup ecosystem, and
  investment environment. Because there are only 50 observations, State should
  be treated as an auxiliary feature.
- Machine learning expert: Because the dataset is small, evaluation should use
  both train-test split and 5-fold cross-validation. The final model should be
  selected using predictive performance, stability, simplicity, and
  interpretability.

Important interpretation rule:
This is observational data with only 50 rows. Use predictive association
language, not causal language. The model can identify patterns associated with
Profit, but it cannot prove that a feature directly causes Profit to increase.
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

    print(
        """

Feature meaning from expert interpretation:
- R&D Spend: Product innovation, technical capability, product development, and
  long-term competitiveness.
- Marketing Spend: Brand exposure, customer acquisition, market expansion, and
  advertising efficiency.
- Administration: Operating cost, company scale, management structure, and
  operational maturity.
- State: Regional business environment, labor cost, tax policy, talent density,
  startup ecosystem, and investment environment.
- Profit: The numeric target to be predicted.
""".rstrip()
    )


def feature_analysis():
    """Print expert-level feature analysis without causal overclaiming."""
    print_section("Expert Feature Analysis and Feature Ranking")
    print(
        """
1. R&D Spend
- Role: Core innovation factor
- Expected importance: High
- Recommendation: Always keep
- Interpretation: R&D Spend is expected to be the strongest predictor because
  it reflects product development, innovation capability, technical
  competitiveness, and long-term growth potential.

2. Marketing Spend
- Role: Market expansion and customer acquisition factor
- Expected importance: Medium to high
- Recommendation: Keep, but check correlation with R&D Spend
- Interpretation: Marketing Spend may help prediction by increasing market
  exposure and customer acquisition. It may also overlap with company size and
  R&D Spend, so avoid interpreting it as an independent causal factor.

3. Administration
- Role: Operating cost and company scale factor
- Expected importance: Low to medium
- Recommendation: Keep first, then evaluate through model comparison
- Interpretation: Administration may be weaker because it does not directly
  create revenue, but it may reflect company scale, management structure, and
  operational maturity.

4. State
- Role: Regional auxiliary factor
- Expected importance: Low to medium
- Recommendation: Use One-Hot Encoding and avoid overinterpretation
- Interpretation: State may reflect regional conditions, but each State has
  limited samples in this small dataset, so it should be treated as auxiliary.
""".strip()
    )


def machine_learning_warnings():
    """Print warnings that guide responsible modeling and interpretation."""
    print_section("Machine Learning Expert Warnings")
    print(
        """
Small sample size:
- Issue: The dataset contains only 50 rows.
- Action: Use 5-fold cross-validation, inspect both mean and standard deviation,
  and avoid strong conclusions from a single train-test split.

Multicollinearity:
- Issue: R&D Spend and Marketing Spend may be correlated.
- Action: Check the correlation matrix and avoid overinterpreting individual
  Linear Regression coefficients. Ridge or Lasso could be tested in future
  versions.

Causality:
- Issue: This is observational data.
- Action: Use predictive association language. Do not claim that one feature
  directly causes Profit to increase.

Model complexity:
- Issue: More features do not always mean a better model.
- Action: Compare feature sets and prefer a stable, interpretable, simpler model
  when performance is similar.
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
            "purpose": "Check the predictive power of the core innovation feature.",
            "expert_question": "How much Profit can be explained by R&D Spend alone?",
        },
        "R&D + Marketing": {
            "features": ["R&D Spend", "Marketing Spend"],
            "purpose": "Check whether Marketing adds value beyond R&D.",
            "expert_question": (
                "Does market expansion add predictive value after product innovation?"
            ),
        },
        "Numerical Features": {
            "features": ["R&D Spend", "Marketing Spend", "Administration"],
            "purpose": "Check whether Administration improves prediction.",
            "expert_question": (
                "Does operating cost or company scale add useful information?"
            ),
        },
        "All Features": {
            "features": [
                "R&D Spend",
                "Marketing Spend",
                "Administration",
                "State",
            ],
            "purpose": "Check whether State improves prediction.",
            "expert_question": "Does regional information add useful predictive value?",
        },
    }

    print("Target variable: Profit")
    print("Feature candidates: R&D Spend, Administration, Marketing Spend, State")
    print("Train-test split: test_size=0.2, random_state=42")
    print("Categorical encoding for State: OneHotEncoder(drop='first')")
    print("Pipeline: ColumnTransformer preprocessing + LinearRegression model")

    print_section("CRISP-DM Step 4: Modeling")
    print(
        """
Primary model: LinearRegression from scikit-learn
Reason:
- Profit is a continuous target.
- The dataset is small.
- Linear Regression is interpretable.
- It is suitable for CRISP-DM teaching and reporting.
""".strip()
    )

    print_section("CRISP-DM Step 5: Evaluation")

    results = []

    for model_name, experiment in model_experiments.items():
        feature_columns = experiment["features"]
        X = df[feature_columns]

        print(f"\nRunning experiment: {model_name}")
        print(f"Purpose: {experiment['purpose']}")
        print(f"Expert question: {experiment['expert_question']}")

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

    return results_df, model_experiments


def _metric_for(results_df, model_name, metric_name):
    """Read a metric value for one model from the comparison table."""
    return float(results_df.loc[results_df["Model"] == model_name, metric_name].iloc[0])


def select_final_model(results_df, model_experiments):
    """Select best model based on CV R2 Mean, stability, and interpretability."""
    ranked_results = results_df.sort_values(
        by=["CV R2 Mean", "CV RMSE Mean", "CV R2 Std"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    best_row = ranked_results.iloc[0]
    best_model_name = best_row["Model"]
    best_features = model_experiments[best_model_name]["features"]

    print_section("Final Model Selection")
    print("Selection rule:")
    print("- Prefer strong CV R2 Mean.")
    print("- Prefer lower CV RMSE Mean.")
    print("- Check CV standard deviation for stability.")
    print("- If models are close, prefer the simpler and more interpretable model.")
    print("- Do not automatically choose the most complex model.")

    print(f"\nSelected final model: {best_model_name}")
    print(f"Selected features: {best_features}")

    rd_only_cv = _metric_for(results_df, "R&D Only", "CV R2 Mean")
    rd_marketing_cv = _metric_for(results_df, "R&D + Marketing", "CV R2 Mean")
    numerical_cv = _metric_for(results_df, "Numerical Features", "CV R2 Mean")
    all_features_cv = _metric_for(results_df, "All Features", "CV R2 Mean")

    marketing_gain = rd_marketing_cv - rd_only_cv
    administration_gain = numerical_cv - rd_marketing_cv
    state_gain = all_features_cv - numerical_cv
    final_gain_over_rd = float(best_row["CV R2 Mean"]) - rd_only_cv

    print("\nModel comparison interpretation:")
    print(
        f"- Marketing gain beyond R&D based on CV R2 Mean: {marketing_gain:.4f}. "
        "A small value suggests Marketing adds limited predictive value beyond "
        "the core R&D feature."
    )
    print(
        f"- Administration gain beyond R&D + Marketing: {administration_gain:.4f}. "
        "A small or negative value suggests Administration is not strongly "
        "improving prediction in this dataset, though it was correctly tested."
    )
    print(
        f"- State gain beyond numerical features: {state_gain:.4f}. "
        "A small or negative value suggests State does not add meaningful "
        "predictive value here."
    )
    print(
        f"- Final model gain over R&D-only model: {final_gain_over_rd:.4f}. "
        "If this value is small, R&D Spend dominates the prediction."
    )

    print(
        """

Final interpretation:
The model comparison shows that R&D Spend is expected to be the most important
predictor of Profit because it reflects product innovation and technical
capability. Marketing Spend may provide additional predictive value by
supporting market expansion, but its effect should be interpreted carefully
because it may be correlated with R&D Spend. Administration may be weaker, but
it can still reflect company scale or operational maturity. State is treated as
an auxiliary categorical feature and encoded using One-Hot Encoding. Because the
dataset contains only 50 observations, the results should be interpreted as
predictive associations rather than causal conclusions. The final model is
selected based on performance, stability, simplicity, and interpretability.
""".rstrip()
    )

    return best_model_name, best_features


def deployment_simulation(final_pipeline):
    """CRISP-DM Step 6: Predict Profit for a new startup."""
    print_section("CRISP-DM Step 6: Deployment Simulation")
    print(
        "This is a learning-project deployment simulation, not a full production "
        "deployment. A production system would need monitoring, data validation, "
        "retraining strategy, access control, and business review."
    )

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
    machine_learning_warnings()

    results_df, model_experiments = run_model_experiments(df)
    best_model_name, best_features = select_final_model(results_df, model_experiments)

    final_pipeline = build_pipeline(best_features)
    final_pipeline.fit(df[best_features], df[TARGET_COLUMN])

    deployment_simulation(final_pipeline)
    save_model(final_pipeline)

    print_section("Workflow Complete")
    print(f"The full CRISP-DM workflow has finished successfully using {best_model_name}.")


if __name__ == "__main__":
    main()
