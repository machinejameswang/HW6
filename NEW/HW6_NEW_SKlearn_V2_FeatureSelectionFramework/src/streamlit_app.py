import streamlit as st
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
