import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

# Setup paths relative to this script in the Final AI Engineer folder
ROOT = Path(__file__).resolve().parent
REG_MODEL_PATH = ROOT / "HW6_NEW_SKlearn_V2_FeatureSelectionFramework" / "outputs" / "models" / "best_regression_pipeline.joblib"
CLF_MODEL_PATH = ROOT / "HW6_NEW_SKlearn_V2_FeatureSelectionFramework" / "outputs" / "models" / "profit_level_classifier_pipeline.joblib"
VOTING_PATH = ROOT / "HW6_NEW_SKlearn_V2_FeatureSelectionFramework" / "outputs" / "feature_selection_voting_matrix.csv"
BENCHMARK_PATH = ROOT / "HW6_NEW_SKlearn_V2_FeatureSelectionFramework" / "outputs" / "regression_benchmark_filter_wrapper_embedded.csv"
MSE_CHART_PATH = ROOT / "feature_selection_performance_allione.png"
INFOGRAPHIC_PATH = ROOT / "one_page_crispdm_feature_selection_infographic.png"

# Page Configuration
st.set_page_config(page_title="Startup Profit AI Infographic", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Infographic Style
st.markdown("""
<style>
    /* Global Styles for Light Infographic Theme */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif !important;
        background-color: #F4F6F9 !important;
        color: #2C3E50 !important;
    }
    
    .stApp {
        background-color: #F4F6F9 !important;
    }

    /* Main Title mimicking the top header in the image */
    .main-header {
        text-align: center;
        background-color: #1A365D;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-size: 28px;
        font-weight: 900;
        letter-spacing: 2px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #1A365D;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 30px;
        border-bottom: 2px solid #CBD5E0;
        padding-bottom: 10px;
    }

    /* Info Cards mimicking the numbered boxes */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border: 2px solid #E2E8F0;
    }
    
    /* Card specific border colors */
    .card-navy { border-color: #1A365D; }
    .card-blue { border-color: #2B6CB0; }
    .card-green { border-color: #2F855A; }
    .card-purple { border-color: #6B46C1; }
    .card-orange { border-color: #C05621; }

    /* Card Headers */
    .card-title {
        display: flex;
        align-items: center;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #E2E8F0;
    }
    
    .card-title .number-badge {
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-size: 18px;
    }
    
    /* Specific Badge Colors */
    .badge-navy { background-color: #1A365D; }
    .card-title.navy { color: #1A365D; border-bottom-color: #1A365D; }
    
    .badge-blue { background-color: #2B6CB0; }
    .card-title.blue { color: #2B6CB0; border-bottom-color: #2B6CB0; }
    
    .badge-green { background-color: #2F855A; }
    .card-title.green { color: #2F855A; border-bottom-color: #2F855A; }
    
    .badge-purple { background-color: #6B46C1; }
    .card-title.purple { color: #6B46C1; border-bottom-color: #6B46C1; }
    
    .badge-orange { background-color: #C05621; }
    .card-title.orange { color: #C05621; border-bottom-color: #C05621; }

    /* Fix Streamlit widgets to match light theme */
    .stSlider > div > div > div > div { background-color: #2B6CB0 !important; }
    .stButton > button {
        background-color: #C05621;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 6px rgba(192, 86, 33, 0.3);
    }
    .stButton > button:hover {
        background-color: #9C4221;
        color: white;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #2F855A;
        font-weight: 900;
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)

# Top Header
st.markdown("<div class='main-header'>AI 機器學習專案 | 10大演算法效能分析</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Startup Profit Feature Selection 資訊圖表</div>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    reg = joblib.load(REG_MODEL_PATH) if REG_MODEL_PATH.exists() else None
    clf = joblib.load(CLF_MODEL_PATH) if CLF_MODEL_PATH.exists() else None
    return reg, clf

reg_model, clf_model = load_models()

# Layout: Two Columns mimicking the A4 split layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # --- CARD 1: Prediction Engine (Navy) ---
    st.markdown("""
    <div class='info-card card-navy'>
        <div class='card-title navy'><span class='number-badge badge-navy'>1</span> 預測引擎 Prediction Engine</div>
    """, unsafe_allow_html=True)
    
    st.markdown("調整下方滑桿，即時預測 Startup 獲利。")
    rd = st.slider("🧪 研發支出 R&D Spend", min_value=0, max_value=500000, value=120000, step=1000)
    marketing = st.slider("📈 行銷支出 Marketing Spend", min_value=0, max_value=1000000, value=250000, step=1000)
    admin = st.slider("🏢 行政支出 Administration", min_value=0, max_value=500000, value=100000, step=1000)
    state = st.selectbox("📍 州別 State Location", ["California", "Florida", "New York"])
    
    if st.button("🚀 執行預測 (Run Prediction)"):
        if reg_model and clf_model:
            x_full = pd.DataFrame([{"R&D Spend": rd, "Administration": admin, "Marketing Spend": marketing, "State": state}])
            x_selected = x_full[["R&D Spend", "Marketing Spend"]]
            
            profit_pred = reg_model.predict(x_selected)[0]
            level_pred = clf_model.predict(x_full)[0]
            
            c1, c2 = st.columns(2)
            c1.metric("預測獲利金額", f"${profit_pred:,.0f}")
            c2.metric("獲利等級", level_pred)
        else:
            st.error("Model not found.")
            
    st.markdown("</div>", unsafe_allow_html=True)

    # --- CARD 3: Regression Benchmark (Green) ---
    st.markdown("""
    <div class='info-card card-green'>
        <div class='card-title green'><span class='number-badge badge-green'>3</span> 回歸模型效能 Benchmark</div>
    """, unsafe_allow_html=True)
    
    if BENCHMARK_PATH.exists():
        bench_df = pd.read_csv(BENCHMARK_PATH)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Benchmark CSV not found.")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- CARD 5: Feature Selection Insights (Orange) ---
    st.markdown("""
    <div class='info-card card-orange'>
        <div class='card-title orange'><span class='number-badge badge-orange'>5</span> 特徵選擇矩陣 Voting Matrix</div>
    """, unsafe_allow_html=True)
    
    if VOTING_PATH.exists():
        voting_df = pd.read_csv(VOTING_PATH)
        st.dataframe(voting_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Voting Matrix CSV not found.")
        
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # --- CARD 2: Feature Selection Process (Light Blue) ---
    st.markdown("""
    <div class='info-card card-blue'>
        <div class='card-title blue'><span class='number-badge badge-blue'>2</span> 特徵選擇流程架構</div>
    """, unsafe_allow_html=True)
    
    if INFOGRAPHIC_PATH.exists():
        st.image(Image.open(INFOGRAPHIC_PATH), use_container_width=True)
    else:
        st.warning("Infographic not found.")
        
    st.markdown("</div>", unsafe_allow_html=True)

    # --- CARD 4: MSE Performance (Purple) ---
    st.markdown("""
    <div class='info-card card-purple'>
        <div class='card-title purple'><span class='number-badge badge-purple'>4</span> MSE 收斂效能圖表</div>
    """, unsafe_allow_html=True)
    
    st.markdown("比較不同特徵演算法在增加特徵數量時的 MSE 變化：")
    if MSE_CHART_PATH.exists():
        st.image(Image.open(MSE_CHART_PATH), use_container_width=True)
    else:
        st.warning("MSE Chart not found.")
        
    st.markdown("</div>", unsafe_allow_html=True)

# Custom JS to force light theme for Streamlit elements if needed
st.markdown("""
<script>
    const elements = window.parent.document.querySelectorAll('.stApp');
    elements[0].style.backgroundColor = '#F4F6F9';
</script>
""", unsafe_allow_html=True)
