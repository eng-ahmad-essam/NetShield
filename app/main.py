import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# إضافة المسار الرئيسي للمشروع لاستيراد الـ Pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import NetShieldPipeline

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="NetShield | Network Intrusion Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# GLOBAL STYLES
# ------------------------------------------
# Design language: a night-shift Network Operations Center console rather
# than a generic SaaS dashboard. Deep instrument-panel charcoal, a warm
# amber "dial" accent for primary actions/controls, and a cool teal
# "trace" accent standing in for a clean signal on an oscilloscope.
# Danger reads as a warm flare-red, never the default pure #F00.
#
# Native Streamlit widgets (sliders, selects, the dataframe grid, radio
# dots, buttons) are themed via .streamlit/config.toml — that's the
# reliable way to recolor BaseWeb/glide-data-grid internals. The CSS
# below only builds the bespoke console chrome around them.
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    :root {
        --bg: #0D1117;
        --bg-raised: #10151D;
        --panel: #151B25;
        --panel-hover: #1A212C;
        --border: #262E3B;
        --border-soft: #1D2430;
        --text-primary: #EDEFF3;
        --text-secondary: #8992A6;
        --text-muted: #5B6577;

        --accent: #E8A33D;          /* amber dial — primary / Tier 1 */
        --accent-strong: #C9832A;
        --accent-soft: rgba(232, 163, 61, 0.12);

        --trace: #45C4B0;           /* teal trace — safe / Tier 2 */
        --trace-soft: rgba(69, 196, 176, 0.12);

        --alert: #E4572E;           /* flare red — danger */
        --alert-soft: rgba(228, 87, 46, 0.13);
    }

    .stApp {
        background-color: var(--bg);
        background-image:
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 34px 34px;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background-color: transparent;
        box-shadow: none;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }

    * { scrollbar-width: thin; scrollbar-color: var(--border) transparent; }

    /* ---------- Console top bar ---------- */
    .ns-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--bg-raised);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 0;
    }
    .ns-topbar-left { display: flex; align-items: center; gap: 14px; }
    .ns-logo-mark {
        width: 38px; height: 38px;
        border-radius: 9px;
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        transform: rotate(45deg);
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .ns-logo-mark span {
        transform: rotate(-45deg);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 14px; color: #14100A;
    }
    .ns-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 18px; font-weight: 700; color: var(--text-primary);
        letter-spacing: 0.03em; line-height: 1.1;
    }
    .ns-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px; color: var(--text-muted);
        margin-top: 3px; letter-spacing: 0.02em;
    }

    .ns-status {
        display: flex; align-items: center; gap: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px; font-weight: 600; color: var(--trace);
        background: var(--trace-soft);
        border: 1px solid rgba(69,196,176,0.35);
        padding: 7px 13px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    .ns-status-dot {
        width: 7px; height: 7px; border-radius: 50%; background: var(--trace);
        box-shadow: 0 0 0 0 rgba(69,196,176,0.6);
        animation: ns-pulse 2s infinite;
    }
    @keyframes ns-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(69,196,176,0.55); }
        70%  { box-shadow: 0 0 0 7px rgba(69,196,176,0); }
        100% { box-shadow: 0 0 0 0 rgba(69,196,176,0); }
    }

    /* ---------- Signature waveform divider ---------- */
    .ns-wave { width: 100%; line-height: 0; margin: 6px 0 20px 0; opacity: 0.65; }

    /* ---------- Section headers ---------- */
    .ns-h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 14px;
        font-weight: 700;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 2px 0 3px 0;
        border-left: 3px solid var(--accent);
        padding-left: 10px;
    }
    .ns-h2-sub {
        color: var(--text-secondary);
        font-size: 13px;
        margin: 0 0 16px 13px;
    }

    /* ---------- Metric cards ---------- */
    .ns-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
        border-top: 3px solid var(--edge, var(--accent));
    }
    .ns-card-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 9px;
    }
    .ns-card-value {
        font-size: 27px;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'IBM Plex Mono', monospace;
    }
    .ns-card-delta {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 5px;
    }

    /* ---------- Generic panel ---------- */
    .ns-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 18px;
    }
    .ns-panel-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 13px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-soft);
    }
    .ns-img-frame {
        background: #F5F6F8;
        border-radius: 8px;
        padding: 10px;
    }

    /* ---------- Alert boxes ---------- */
    .ns-alert-danger {
        background: var(--alert-soft);
        border: 1px solid rgba(228,87,46,0.4);
        border-left: 4px solid var(--alert);
        border-radius: 6px;
        padding: 13px 16px;
        color: #FFC4B0;
        font-weight: 600;
        font-size: 13.5px;
    }
    .ns-alert-safe {
        background: var(--trace-soft);
        border: 1px solid rgba(69,196,176,0.4);
        border-left: 4px solid var(--trace);
        border-radius: 6px;
        padding: 13px 16px;
        color: #BFF0E7;
        font-weight: 600;
        font-size: 13.5px;
    }
    .ns-alert-neutral {
        background: var(--bg-raised);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 13px 16px;
        color: var(--text-secondary);
        font-size: 13.5px;
    }

    /* ---------- Verdict card (Custom Test) ---------- */
    .ns-verdict {
        border-radius: 12px;
        padding: 26px 28px;
        border: 1px solid var(--border);
    }
    .ns-verdict.attack { background: var(--alert-soft); border-color: rgba(228,87,46,0.45); }
    .ns-verdict.safe { background: var(--trace-soft); border-color: rgba(69,196,176,0.45); }
    .ns-verdict-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
        color: var(--text-secondary); margin-bottom: 6px;
    }
    .ns-verdict-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px; font-weight: 700; color: var(--text-primary);
    }
    .ns-verdict-sub { font-size: 13.5px; color: var(--text-secondary); margin-top: 8px; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-raised);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.3rem; }
    section[data-testid="stSidebar"] * { color: var(--text-primary); }

    .ns-side-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 16px;
        margin-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }
    .ns-side-logo-text {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 15px; color: var(--text-primary);
        letter-spacing: 0.02em;
    }
    .ns-side-logo-sub { font-size: 11px; color: var(--text-muted); margin-top: -1px; }

    .ns-side-caption {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-weight: 700;
        margin: 6px 0 8px 2px;
    }

    /* Turn the nav radio into an unmistakable vertical tab strip */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex; flex-direction: column; gap: 6px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 8px;
        padding: 10px 12px !important;
        margin: 0 !important;
        transition: all .12s ease;
        width: 100%;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        border-color: var(--accent);
        background: var(--accent-soft);
        box-shadow: inset 3px 0 0 var(--accent);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 13.5px !important; font-weight: 600;
    }

    /* Table */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid var(--border);
        overflow: hidden;
    }

    /* Buttons */
    .stButton>button {
        background: var(--accent);
        color: #14100A;
        font-weight: 700;
        font-size: 13.5px;
        border: 1px solid var(--accent);
        border-radius: 7px;
        padding: 9px 16px;
        box-shadow: none;
        transition: background .12s ease;
    }
    .stButton>button:hover {
        background: var(--accent-strong);
        border-color: var(--accent-strong);
        color: #14100A;
    }
    .stDownloadButton>button {
        background: transparent;
        color: var(--trace);
        border: 1px solid var(--trace);
        border-radius: 7px;
        font-weight: 600;
        font-size: 13px;
    }
    .stDownloadButton>button:hover {
        background: var(--trace-soft);
    }

    hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

WAVEFORM_SVG = """
<div class="ns-wave">
<svg viewBox="0 0 1200 46" width="100%" height="46" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <polyline fill="none" stroke="#45C4B0" stroke-width="1.6"
    points="0,23 60,23 80,8 100,38 120,23 220,23 240,14 255,32 270,23 400,23
            420,4 440,40 460,23 600,23 615,16 630,30 645,23 780,23 800,6 820,40 840,23
            960,23 980,15 995,31 1010,23 1200,23" />
</svg>
</div>
"""

# ==========================================
# PIPELINE LOADING
# ==========================================
@st.cache_resource
def load_pipeline():
    return NetShieldPipeline()

@st.cache_data
def load_dataset(path):
    return pd.read_parquet(path)

with st.spinner("Loading detection models..."):
    pipeline = load_pipeline()

PROCESSED_PATH = "data/processed/cleaned_unsw_nb15.parquet"


# ==========================================
# SIDEBAR — navigation lives here, top-left, always visible
# ==========================================
with st.sidebar:
    st.markdown("""
        <div class="ns-side-logo">
            <div class="ns-logo-mark" style="width:32px;height:32px;">
                <span style="font-size:12px;">NS</span>
            </div>
            <div>
                <div class="ns-side-logo-text">NetShield</div>
                <div class="ns-side-logo-sub">Intrusion Detection</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ns-side-caption">◆ Switch view</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["📊  Overview & Metrics", "📡  Batch Inspector", "🧪  Custom Packet Test"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ns-side-caption">◆ System</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="ns-status" style="width:100%; box-sizing:border-box; justify-content:flex-start;">
            <div class="ns-status-dot"></div> Models loaded
        </div>
    """, unsafe_allow_html=True)

    with st.expander("About this pipeline"):
        st.caption(
            "Two-tier ML pipeline trained on UNSW-NB15. Tier 1 separates "
            "normal from attack traffic; Tier 2 classifies attacks into "
            "one of nine categories."
        )
    st.caption("v1.3.0")


# ==========================================
# TOP BAR (shared across pages)
# ==========================================
PAGE_META = {
    "📊  Overview & Metrics": ("System Overview & Metrics", "PIPELINE // EVALUATION SUMMARY"),
    "📡  Batch Inspector": ("Batch Traffic Inspector", "SAMPLE // TIER 1 + TIER 2 CLASSIFICATION"),
    "🧪  Custom Packet Test": ("Custom Packet Test", "MANUAL INPUT // LIVE SCORING"),
}
page_title, page_kicker = PAGE_META[page]

st.markdown(f"""
    <div class="ns-topbar">
        <div class="ns-topbar-left">
            <div class="ns-logo-mark"><span>NS</span></div>
            <div>
                <div class="ns-title">{page_title}</div>
                <div class="ns-subtitle">{page_kicker}</div>
            </div>
        </div>
        <div class="ns-status"><div class="ns-status-dot"></div> Engine Online</div>
    </div>
""", unsafe_allow_html=True)
st.markdown(WAVEFORM_SVG, unsafe_allow_html=True)


def metric_card(label, value, delta, col, edge="var(--accent)"):
    col.markdown(f"""
        <div class="ns-card" style="--edge:{edge};">
            <div class="ns-card-label">{label}</div>
            <div class="ns-card-value">{value}</div>
            <div class="ns-card-delta">{delta}</div>
        </div>
    """, unsafe_allow_html=True)


def plotly_dark_layout(fig, height=340):
    fig.update_layout(
        showlegend=False,
        margin=dict(t=6, b=6, l=6, r=24),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#EDEFF3', family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='#262E3B', zeroline=False, title=None),
        yaxis=dict(title=None),
        height=height,
        bargap=0.35,
    )
    return fig


# ==========================================
# PAGE 1 — OVERVIEW & METRICS
# ==========================================
if page == "📊  Overview & Metrics":
    st.markdown('<div class="ns-h2">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="ns-h2-sub">Evaluation results across the hierarchical pipeline stages</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    metric_card("Tier 1 Accuracy", "99.25%", "Normal vs. Attack", c1, edge="var(--accent)")
    metric_card("Tier 2 Accuracy", "81.04%", "9 Attack Classes", c2, edge="var(--trace)")
    metric_card("End-to-End Accuracy", "97.89%", "Full Pipeline", c3, edge="linear-gradient(90deg, var(--accent), var(--trace))")

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ns-h2">Confusion Matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="ns-h2-sub">Per-tier classification breakdown on the held-out test set</div>', unsafe_allow_html=True)

    t1_cm_path = "results/figures/tier1_confusion_matrix.png"
    t2_cm_path = "results/figures/tier2_confusion_matrix.png"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="ns-panel"><div class="ns-panel-title">Tier 1 — Binary Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists(t1_cm_path):
            st.markdown('<div class="ns-img-frame">', unsafe_allow_html=True)
            st.image(t1_cm_path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ns-alert-neutral">Run Phase 5 to generate the Tier 1 plot.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ns-panel"><div class="ns-panel-title">Tier 2 — Multiclass Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists(t2_cm_path):
            st.markdown('<div class="ns-img-frame">', unsafe_allow_html=True)
            st.image(t2_cm_path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ns-alert-neutral">Run Phase 6 to generate the Tier 2 plot.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 2 — BATCH INSPECTOR (was "Live Traffic Inspector")
# ==========================================
elif page == "📡  Batch Inspector":
    st.markdown('<div class="ns-h2">Batch Traffic Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="ns-h2-sub">Sample packets from the processed dataset and classify them through Tier 1 &amp; Tier 2</div>', unsafe_allow_html=True)

    if not os.path.exists(PROCESSED_PATH):
        st.markdown('<div class="ns-alert-danger">Cleaned dataset not found. Run Phase 4 first to generate it.</div>', unsafe_allow_html=True)
    else:
        df = load_dataset(PROCESSED_PATH)

        st.markdown('<div class="ns-panel">', unsafe_allow_html=True)
        ctrl1, ctrl2 = st.columns([3, 1])
        with ctrl1:
            sample_size = st.slider("Number of random packets to inspect", 5, 50, 10)
        with ctrl2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run = st.button("Analyze Traffic", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        if run:
            sample_df = df.sample(n=sample_size, random_state=np.random.randint(0, 10000)).reset_index(drop=True)

            with st.spinner("Classifying packets via Tier 1 & Tier 2..."):
                final_preds, t1_preds, t1_probas = pipeline.predict(sample_df)

            results_df = pd.DataFrame({
                'Packet ID': sample_df.index + 1,
                'Tier 1 Result': ['Attack' if p == 1 else 'Normal' for p in t1_preds],
                'Attack Probability': [round(prob * 100, 1) for prob in t1_probas],
                'Final Prediction': final_preds,
                'Actual Category': sample_df['attack_cat'],
            })

            threat_count = int((final_preds != 'Normal').sum())
            safe_count = sample_size - threat_count

            # ---- Summary row ----
            s1, s2, s3 = st.columns(3)
            metric_card("Packets Analyzed", str(sample_size), "This batch", s1, edge="var(--border)")
            metric_card("Threats Detected", str(threat_count), "Flagged as attack", s2, edge="var(--alert)")
            metric_card("Clean Traffic", str(safe_count), "Classified normal", s3, edge="var(--trace)")

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if threat_count > 0:
                st.markdown(
                    f'<div class="ns-alert-danger">Security Alert — {threat_count} potential attack vector(s) '
                    f'detected out of {sample_size} packets analyzed.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ns-alert-safe">All network packets evaluated as safe — no anomalies detected.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

            # ---- Results table + breakdown chart ----
            tbl_col, chart_col = st.columns([2, 1])

            with tbl_col:
                st.markdown('<div class="ns-panel-title" style="border:none;padding-bottom:0;">Classification Results</div>', unsafe_allow_html=True)

                def highlight_attack(row):
                    color = 'background-color: rgba(228,87,46,0.18)' if row['Final Prediction'] != 'Normal' else ''
                    return [color] * len(row)

                styled = results_df.style.apply(highlight_attack, axis=1).format({'Attack Probability': '{:.1f}%'})
                st.dataframe(styled, use_container_width=True, height=380)

                st.download_button(
                    "Download results as CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="netshield_batch_results.csv",
                    mime="text/csv",
                )

            with chart_col:
                st.markdown('<div class="ns-panel-title" style="border:none;padding-bottom:0;">Prediction Breakdown</div>', unsafe_allow_html=True)

                pred_counts = results_df['Final Prediction'].value_counts().reset_index()
                pred_counts.columns = ['Category', 'Count']

                # Semantic palette: Normal reads cool/safe (teal trace),
                # every attack category reads warm (amber → flare-red),
                # instead of an arbitrary blue/gray cycle.
                warm_cycle = ["#E4572E", "#E8A33D", "#C9832A", "#B23A1E", "#F08A5D", "#D96C2B", "#8C2E17", "#F2B15A"]
                color_map = {}
                fallback_idx = 0
                for cat in pred_counts['Category']:
                    if cat == 'Normal':
                        color_map[cat] = "#45C4B0"
                    else:
                        color_map[cat] = warm_cycle[fallback_idx % len(warm_cycle)]
                        fallback_idx += 1

                fig = px.bar(
                    pred_counts.sort_values('Count', ascending=True),
                    x='Count', y='Category', orientation='h',
                    color='Category', color_discrete_map=color_map,
                    text='Count',
                )
                fig.update_traces(textposition='outside', textfont=dict(color='#EDEFF3', size=12))
                fig = plotly_dark_layout(fig, height=340)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="ns-alert-neutral">Set a sample size and click <b>Analyze Traffic</b> to run packets through the pipeline.</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 3 — CUSTOM PACKET TEST (new)
# ------------------------------------------
# Lets an analyst seed a single real packet, hand-edit its feature
# values, and push that exact record through Tier 1 + Tier 2 — a
# sandbox for "what if this value were different" testing, without
# needing to hardcode the dataset's full feature schema anywhere.
# ==========================================
elif page == "🧪  Custom Packet Test":
    st.markdown('<div class="ns-h2">Custom Packet Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="ns-h2-sub">Seed a real packet, edit any feature by hand, and score it through the live pipeline</div>', unsafe_allow_html=True)

    if not os.path.exists(PROCESSED_PATH):
        st.markdown('<div class="ns-alert-danger">Cleaned dataset not found. Run Phase 4 first to generate it.</div>', unsafe_allow_html=True)
    else:
        df = load_dataset(PROCESSED_PATH)
        target_cols = [c for c in ['attack_cat', 'label', 'id'] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in target_cols]

        attack_categories = sorted([c for c in df['attack_cat'].unique() if c != 'Normal']) if 'attack_cat' in df.columns else []

        st.session_state.setdefault('custom_seed_counter', 0)
        st.session_state.setdefault('custom_base_row', None)

        st.markdown('<div class="ns-panel">', unsafe_allow_html=True)
        seed_col1, seed_col2, seed_col3 = st.columns([1.4, 1.4, 1])
        with seed_col1:
            seed_choice = st.selectbox(
                "Seed the test from",
                ["Random Normal packet", "Random Attack packet"] + [f"Random '{c}' packet" for c in attack_categories],
            )
        with seed_col2:
            st.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)
        with seed_col3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            load_clicked = st.button("Load Packet", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if load_clicked:
            if seed_choice == "Random Normal packet" and 'attack_cat' in df.columns:
                pool = df[df['attack_cat'] == 'Normal']
            elif seed_choice == "Random Attack packet" and 'attack_cat' in df.columns:
                pool = df[df['attack_cat'] != 'Normal']
            elif 'attack_cat' in df.columns:
                cat = seed_choice.split("'")[1]
                pool = df[df['attack_cat'] == cat]
            else:
                pool = df

            if len(pool) == 0:
                pool = df

            st.session_state['custom_base_row'] = pool.sample(n=1, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
            st.session_state['custom_seed_counter'] += 1

        base_row = st.session_state['custom_base_row']

        if base_row is None:
            st.markdown('<div class="ns-alert-neutral">Choose a seed above and click <b>Load Packet</b> to start editing a real record.</div>', unsafe_allow_html=True)
        else:
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            edit_col, verdict_col = st.columns([1.6, 1])

            with edit_col:
                st.markdown('<div class="ns-panel-title" style="border:none;padding-bottom:0;">Edit Feature Values</div>', unsafe_allow_html=True)
                if 'attack_cat' in base_row.columns:
                    st.caption(f"Seeded from an actual **{base_row['attack_cat'].iloc[0]}** packet — edit any value, then run the test.")

                edit_source = base_row[feature_cols].T.reset_index()
                edit_source.columns = ['Feature', 'Value']
                edit_source['Value'] = edit_source['Value'].astype(str)

                edited = st.data_editor(
                    edit_source,
                    hide_index=True,
                    use_container_width=True,
                    height=460,
                    key=f"custom_editor_{st.session_state['custom_seed_counter']}",
                    column_config={
                        "Feature": st.column_config.TextColumn("Feature", disabled=True),
                        "Value": st.column_config.TextColumn("Value"),
                    },
                )

                run_test = st.button("Run Custom Test", use_container_width=True)

            with verdict_col:
                st.markdown('<div class="ns-panel-title" style="border:none;padding-bottom:0;">Verdict</div>', unsafe_allow_html=True)

                if run_test:
                    try:
                        edited_values = edited.set_index('Feature')['Value']
                        new_row = base_row.copy()
                        for feat in feature_cols:
                            orig_dtype = base_row[feat].dtype
                            new_row.at[0, feat] = pd.Series([edited_values[feat]]).astype(orig_dtype).iloc[0]
                    except Exception as exc:
                        st.markdown(
                            f'<div class="ns-alert-danger">Could not parse one of the edited values — check that '
                            f'numeric features contain numbers. ({exc})</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        with st.spinner("Scoring packet..."):
                            final_preds, t1_preds, t1_probas = pipeline.predict(new_row)

                        final_pred = final_preds[0]
                        t1_pred = t1_preds[0]
                        t1_proba = t1_probas[0]
                        is_attack = final_pred != 'Normal'
                        verdict_class = "attack" if is_attack else "safe"

                        st.markdown(f"""
                            <div class="ns-verdict {verdict_class}">
                                <div class="ns-verdict-eyebrow">Final Prediction</div>
                                <div class="ns-verdict-value">{final_pred}</div>
                                <div class="ns-verdict-sub">Tier 1: {'Attack' if t1_pred == 1 else 'Normal'} · Attack probability {t1_proba*100:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)

                        if 'attack_cat' in base_row.columns:
                            original_cat = base_row['attack_cat'].iloc[0]
                            if str(original_cat) != str(final_pred):
                                st.markdown(
                                    f'<div class="ns-alert-neutral" style="margin-top:14px;">Your edits shifted the '
                                    f'classification — the seed packet was originally <b>{original_cat}</b>.</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div class="ns-alert-neutral" style="margin-top:14px;">Your edits did not '
                                    'change the classification versus the original seed packet.</div>',
                                    unsafe_allow_html=True,
                                )
                else:
                    st.markdown(
                        '<div class="ns-alert-neutral">Edit values on the left, then click '
                        '<b>Run Custom Test</b> to score this exact packet.</div>',
                        unsafe_allow_html=True,
                    )
