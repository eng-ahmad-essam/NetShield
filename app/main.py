import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# إضافة المسار الرئيسي للمشروع لاستيراد الـ Pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import NetShieldPipeline

# إعدادات الصفحة
st.set_page_config(
    page_title="NetShield | AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# تحميل الـ Pipeline مرة واحدة لسرعة الأداء
@st.cache_resource
def load_pipeline():
    return NetShieldPipeline()

pipeline = load_pipeline()

# العنوان الرئيسي
st.title("🛡️ NetShield: Hierarchical AI Intrusion Detection System")
st.markdown("An end-to-end multi-tiered threat detection platform for real-time network traffic evaluation.")

# القائمة الجانبية
st.sidebar.header("🕹️ Navigation & Controls")
menu = st.sidebar.radio("Go to:", ["📊 System Dashboard & Metrics", "🔍 Live Traffic Inspector"])

# ==========================================
# PAGE 1: DASHBOARD & METRICS
# ==========================================
if menu == "📊 System Dashboard & Metrics":
    st.header("📈 Architecture & Model Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tier 1 Accuracy", "99.25%", "Normal vs Attack")
    col2.metric("Tier 2 Accuracy", "81.04%", "9 Attack Classes")
    col3.metric("System End-to-End Accuracy", "97.89%", "Full Pipeline")
    
    st.divider()
    
    st.subheader("🖼️ Model Confusion Matrices")
    c1, c2 = st.columns(2)
    
    t1_cm_path = "results/figures/tier1_confusion_matrix.png"
    t2_cm_path = "results/figures/tier2_confusion_matrix.png"
    
    with c1:
        st.write("**Tier 1 Binary Confusion Matrix**")
        if os.path.exists(t1_cm_path):
            st.image(t1_cm_path, use_container_width=True)
        else:
            st.info("Run Phase 5 to generate Tier 1 plot.")
            
    with c2:
        st.write("**Tier 2 Multiclass Confusion Matrix**")
        if os.path.exists(t2_cm_path):
            st.image(t2_cm_path, use_container_width=True)
        else:
            st.info("Run Phase 6 to generate Tier 2 plot.")

# ==========================================
# PAGE 2: LIVE TRAFFIC INSPECTOR
# ==========================================
elif menu == "🔍 Live Traffic Inspector":
    st.header("⚡ Live Network Traffic Inspection")
    st.markdown("Test random network samples against the trained hierarchical models.")
    
    processed_path = "data/processed/cleaned_unsw_nb15.parquet"
    if os.path.exists(processed_path):
        df = pd.read_parquet(processed_path)
        
        sample_size = st.slider("Select number of random network packets to inspect:", 5, 50, 10)
        
        if st.button("🚀 Analyze Traffic Samples"):
            sample_df = df.sample(n=sample_size, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
            
            with st.spinner("Classifying packets via Tier 1 & Tier 2..."):
                final_preds, t1_preds, t1_probas = pipeline.predict(sample_df)
                
            results_df = pd.DataFrame({
                'Packet ID': sample_df.index + 1,
                'Tier 1 Result': ['Attack (1)' if p == 1 else 'Normal (0)' for p in t1_preds],
                'Attack Probability': [f"{prob*100:.1f}%" for prob in t1_probas],
                'Final Prediction': final_preds,
                'Actual Category': sample_df['attack_cat']
            })
            
            st.subheader("📋 Classification Results Table")
            st.dataframe(results_df, use_container_width=True)
            
            # ملخص التهديدات
            threat_count = (final_preds != 'Normal').sum()
            if threat_count > 0:
                st.error(f"⚠️ Security Alert: {threat_count} potential attack vector(s) detected out of {sample_size} packets!")
            else:
                st.success("✅ All network packets evaluated as safe (Normal Traffic).")
    else:
        st.warning("Cleaned dataset missing! Run Phase 4 first.")