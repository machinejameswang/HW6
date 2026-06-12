# HW6 NEW SKlearn V2 - Filter / Wrapper / Embedded Feature Selection Framework

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
