import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# ==========================================
# CRISP-DM Step 1: Business Understanding
# ==========================================
# Objective: Develop a multiple linear regression model to predict the profit of a startup 
# based on its expenditures in R&D, Administration, and Marketing, as well as its State.
# This helps venture capitalists understand which factors are most critical for a startup's profitability.

def main():
    # Resolve the path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '50_Startups_dataset.csv')

    print("--- 1. Business Understanding ---")
    print("Objective: Predict startup profit based on R&D, Administration, Marketing, and State.")
    print("Goal: Identify critical factors driving profitability for venture capitalists.\n")

    # ==========================================
    # CRISP-DM Step 2: Data Understanding
    # ==========================================
    # Report:
    # * Dataset contains 50 startup companies.
    # * Target variable: Profit.
    # * Features include R&D Spend, Administration, Marketing Spend, and State.
    # * No missing values were detected.
    # * One categorical feature (State) was identified.
    # * An index column (Unnamed:0) was found and deemed irrelevant for prediction.
    print("--- 2. Data Understanding ---")
    df = pd.read_csv(dataset_path)

    # The dataset has an unnamed index column as the first column, let's remove it if it exists
    if df.columns[0] == 'Unnamed: 0':
        df = df.drop(columns=[df.columns[0]])

    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    # State vs Profit Analysis
    print("\nState vs Profit Analysis:")
    print(df.groupby("State")["Profit"].mean().sort_values(ascending=False))

    print("\nCorrelation Analysis (Numeric Features vs Profit):")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix['Profit'].sort_values(ascending=False))

    # Multicollinearity Check
    print("\nMulticollinearity Check (Correlation among predictors):")
    print(df[['R&D Spend', 'Administration', 'Marketing Spend']].corr())

    # VIF Analysis
    print("\nVariance Inflation Factor (VIF) Analysis:")
    # Calculate VIF for numeric features (R&D, Admin, Marketing)
    # We add a constant to calculate VIF correctly
    X_vif = df[['R&D Spend', 'Administration', 'Marketing Spend']]
    X_vif_const = sm.add_constant(X_vif)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif_const.values, i) for i in range(len(X_vif_const.columns))]
    print(vif_data[vif_data['Feature'] != 'const'])

    # Outlier Analysis using Boxplots
    print("\nPerforming Outlier Analysis (Saving boxplots to 'outliers.png')...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df["Profit"])
    plt.title("Boxplot of Profit")
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df["R&D Spend"])
    plt.title("Boxplot of R&D Spend")
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'outliers.png'))
    plt.close()

    # ==========================================
    # CRISP-DM Step 3: Data Preparation
    # ==========================================
    # Report:
    # * Removed unnecessary index column.
    # * Separated features and target variable.
    # * Applied One-Hot Encoding to State.
    # * Split dataset into training and testing sets (80/20).
    # * Recommended and applied additional preprocessing:
    #   - Feature scaling (StandardScaler)
    #   - Correlation analysis (in Step 2)
    #   - Feature selection (removed 'Administration')
    #   * Note: Log transformation was evaluated but discarded as R&D Spend has a strictly linear relationship with Profit.
    print("\n--- 3. Data Preparation ---")
    
    # (Log transformation was removed here to preserve the strong linear relationship with Profit)

    # Separate features (independent variables) and target (dependent variable)
    X = df.drop(columns=['Profit'])
    y = df['Profit']

    # Feature Selection: Remove 'Administration' due to weak correlation with Profit
    print("Performing Feature Selection: Dropping 'Administration' due to weak correlation...")
    X = X.drop(columns=['Administration'])

    # Identify categorical and numeric features
    categorical_features = ['State']
    numeric_features = ['R&D Spend', 'Marketing Spend']

    # Handle categorical data ('State') using OneHotEncoding
    # and scale numeric features using StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features) 
        ])

    X_processed = preprocessor.fit_transform(X)
    print(f"Shape of processed features (X): {X_processed.shape}")

    # Split the dataset into the Training set (80%) and Test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


    # ==========================================
    # CRISP-DM Step 4: Modeling
    # ==========================================
    print("\n--- 4. Modeling ---")
    # Initialize the Multiple Linear Regression model
    regressor = LinearRegression()
    
    # Train the model using the training data
    regressor.fit(X_train, y_train)
    print("Multiple Linear Regression model trained successfully.")

    # Initialize Random Forest Regressor for Comparison
    print("\nTraining Random Forest Regressor for Comparison...")
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    print("Random Forest Regressor trained successfully.")

    # Cross Validation
    print("\nPerforming 5-Fold Cross Validation on Linear Regression...")
    cv_scores_lr = cross_val_score(regressor, X_processed, y, cv=5, scoring='r2')
    print(f"Linear Regression CV R2 Scores: {cv_scores_lr}")
    print(f"Linear Regression Mean CV R2: {cv_scores_lr.mean():.4f}")


    # ==========================================
    # CRISP-DM Step 5: Evaluation
    # ==========================================
    print("\n--- 5. Evaluation ---")
    # Predict the Test set results for Linear Regression
    y_pred = regressor.predict(X_test)
    y_pred_rf = rf_regressor.predict(X_test)

    # Calculate evaluation metrics for Linear Regression
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate evaluation metrics for Random Forest
    rf_r2 = r2_score(y_test, y_pred_rf)

    print("\n--- Linear Regression Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    print("\n--- Random Forest Performance ---")
    print(f"R-squared (R2): {rf_r2:.4f}")
    print(f"Model Comparison: Linear Regression (R2={r2:.4f}) vs Random Forest (R2={rf_r2:.4f})")

    # Feature Importance Ranking (Using Random Forest)
    print("\nFeature Importance Ranking (from Random Forest):")
    feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    rf_importances = pd.DataFrame({'Feature': feature_names, 'Importance': rf_regressor.feature_importances_})
    print(rf_importances.sort_values(by='Importance', ascending=False))

    # Actual vs Predicted Plot
    print("\nSaving Actual vs Predicted plot to 'actual_vs_predicted.png'...")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted (LR)')
    # Plot perfect prediction line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Profit')
    plt.ylabel('Predicted Profit')
    plt.title('Actual vs Predicted Profit (Linear Regression)')
    plt.legend()
    plt.savefig(os.path.join(script_dir, 'actual_vs_predicted.png'))
    plt.close()

    # Residual Analysis
    print("Performing Residual Analysis (Saving plot to 'residual_plot.png')...")
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Profit')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis (Predicted vs Residuals)')
    plt.savefig(os.path.join(script_dir, 'residual_plot.png'))
    plt.close()


    # ==========================================
    # CRISP-DM Step 6: Deployment
    # ==========================================
    print("\n--- 6. Deployment ---")
    # In a real-world scenario, we deploy the model so it can be consumed by applications.
    # Here, we simulate deployment by saving the trained model and preprocessor to disk.
    
    model_filename = 'multiple_linear_regression_model.joblib'
    preprocessor_filename = 'preprocessor.joblib'
    
    joblib.dump(regressor, os.path.join(script_dir, model_filename))
    joblib.dump(preprocessor, os.path.join(script_dir, preprocessor_filename))
    
    print(f"Model saved to: {model_filename}")
    print(f"Preprocessor saved to: {preprocessor_filename}")
    print("The model is ready to be loaded and used for future predictions in a production environment.")

if __name__ == "__main__":
    main()
