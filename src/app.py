import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DEMO = ROOT / "src" / "models" / "demo.py"
GREEN = "#2E8B57"
GREY = "#9DB8A8"

st.set_page_config(page_title="Financial Literacy Toolkit", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 1150px;}
.hero {background: linear-gradient(120deg, #2E8B57 0%, #5FB380 100%);
       padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {color: #EAF6EE; margin: 0.3rem 0 0 0; font-size: 1.02rem;}
[data-testid="stMetric"] {background: #F3FAF5; border: 1px solid #CFE8D8;
       border-radius: 12px; padding: 0.9rem 1rem;}
.stTabs [data-baseweb="tab"] {font-size: 1.02rem; padding: 0.6rem 1.1rem;}
.note {background: #F3FAF5; border-left: 5px solid #2E8B57; padding: 0.8rem 1rem;
       border-radius: 8px; margin: 0.8rem 0;}
.label-box {text-align: center; border-radius: 14px; padding: 1.4rem;
       background: #EAF6EE; border: 2px solid #2E8B57;}
.label-box h2 {margin: 0; color: #1B3A2A;}
.label-box p {margin: 0.3rem 0 0 0; color: #3E6B52;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
<h1>ML-Based Financial Literacy Toolkit</h1>
<p>Classical machine learning that teaches how market patterns and risk work, and explains its own reasoning.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write("Pattern classifier, risk tiers and SHAP explanations on historical US stock prices.")
    st.markdown(
        "- 394 stocks\n- 1,001,927 labeled days\n- 15 scale-free features\n"
        "- Random Forest, Gradient Boosting, K-Means, SHAP"
    )
    st.info("Educational tool. Not a trading system or investment advice.")
    st.caption("Divya Ohri | B.Tech CSE | Machine Learning course project")


def load_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Label rule", "Live demo", "What it learned", "Risk tiers"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline accuracy", "30.8%")
    c2.metric("Random Forest", "40.2%", "+9.4 pts")
    c3.metric("Gradient Boosting", "40.3%", "+9.5 pts")
    c4.metric("Volatility-scaled edge", "+3.2 pts", "95% CI +1.3 to +5.2")

    left, right = st.columns(2)
    with left:
        st.subheader("Model vs baseline")
        acc = pd.DataFrame(
            {"Accuracy (%)": [30.8, 40.2, 40.3]},
            index=["Baseline", "Random Forest", "Gradient Boosting"],
        )
        st.bar_chart(acc, color=GREEN)
    with right:
        st.subheader("Walk-forward: four test windows")
        wf = pd.DataFrame(
            {"Model": [36.0, 38.6, 40.5, 40.3], "Baseline": [24.5, 30.5, 33.5, 31.4]},
            index=["Window 1", "Window 2", "Window 3", "Window 4"],
        )
        st.bar_chart(wf, color=[GREEN, GREY])
    st.markdown(
        '<div class="note">Split by time: train Mar 1982 to Oct 2015 (798,745 rows), 10-day gap, '
        "test Oct 2015 to Nov 2017 (200,555 rows). My first run scored 88.9%: that was data "
        "leakage. The honest score is about 40%.</div>",
        unsafe_allow_html=True,
    )
    df = load_csv("v2_classification_summary.csv")
    if df is not None:
        with st.expander("Full classification summary (from my run)"):
            st.dataframe(df, width="stretch")
    st.subheader("Honest limits")
    st.markdown(
        "- Most of the signal is volatility, not direction.\n"
        "- Medium and High risk tiers (original method) are unstable; Low is reliable.\n"
        "- Only 394 stocks, test periods within 2009 to 2017, data ends 2017, no trading costs.\n"
        "- Bullish vs Bearish stays hard to predict."
    )

with tab2:
    st.subheader("How a day gets its label")
    st.write("Each day is labeled by what the price does over the next 5 trading days.")
    a, b = st.columns(2)
    r = a.slider("Forward 5-day return (%)", -8.0, 8.0, 1.0, 0.5)
    v = b.slider("Forward volatility / current 20-day volatility", 0.2, 1.6, 0.7, 0.05)
    if r > 3:
        label, why = "Bullish", "Return above +3%"
    elif r < -3:
        label, why = "Bearish", "Return below -3%"
    elif v < 0.8:
        label, why = "Consolidation", "Within 3% and calm (below 0.8x)"
    else:
        label, why = "Neutral", "Within 3% but not calm"
    st.markdown(
        f'<div class="label-box"><h2>{label}</h2><p>{why}</p></div>',
        unsafe_allow_html=True,
    )

with tab3:
    st.subheader("Live demo on unseen test data")
    st.write("Runs the real trained Random Forest on a stock and date it never saw in training. Leave both boxes empty for a random pick.")
    a, b = st.columns(2)
    ticker = a.text_input("Ticker", placeholder="achv.us")
    date = b.text_input("Date", placeholder="2016-07-12")
    if st.button("Run demo", type="primary"):
        if bool(ticker.strip()) != bool(date.strip()):
            st.error("Fill both ticker and date, or leave both empty.")
        elif not DEMO.exists():
            st.error(f"Not found: {DEMO}")
        else:
            cmd = [sys.executable, str(DEMO)]
            if ticker.strip():
                cmd += [ticker.strip(), date.strip()]
            with st.spinner("Loading model and running..."):
                try:
                    p = subprocess.run(
                        cmd, cwd=ROOT, capture_output=True, text=True,
                        timeout=600, encoding="utf-8", errors="replace",
                    )
                    st.code(p.stdout or "(no output)")
                    if p.returncode != 0:
                        st.error("demo.py exited with an error:")
                        st.code(p.stderr)
                except subprocess.TimeoutExpired:
                    st.error("Timed out after 10 minutes.")
    st.markdown(
        '<div class="note">Accuracy is about 40%, so WRONG answers are expected. '
        "High volatility raises the odds of both Bullish and Bearish: it signals a big move, not its direction.</div>",
        unsafe_allow_html=True,
    )

with tab4:
    st.subheader("Which clues the forest relied on (SHAP)")
    fi = load_csv("v2_feature_importance.csv")
    if fi is None:
        st.warning("results/v2_feature_importance.csv not found.")
    else:
        nums = fi.select_dtypes("number").columns.tolist()
        texts = [c for c in fi.columns if c not in nums]
        if nums:
            col = nums[-1]
            idx = texts[0] if texts else fi.columns[0]
            st.bar_chart(fi.set_index(idx)[col].sort_values(ascending=False), color=GREEN)
        with st.expander("Show table"):
            st.dataframe(fi, width="stretch")
    st.markdown(
        '<div class="note">Volatility (0.045) and Bollinger width (0.024) dominate; every other '
        "feature is 0.006 or less. The model mostly learned how wild the market is, not which way it will go.</div>",
        unsafe_allow_html=True,
    )

with tab5:
    st.subheader("Stock risk tiers (K-Means)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Low", "286 stocks", "1.81% daily swing", delta_color="off")
    c2.metric("Medium", "83 stocks", "4.13% daily swing", delta_color="off")
    c3.metric("High", "25 stocks", "5.07% daily swing", delta_color="off")
    rt = load_csv("v2_risk_tiers.csv")
    if rt is None:
        st.warning("results/v2_risk_tiers.csv not found.")
    else:
        q = st.text_input("Search a stock", placeholder="aapl")
        view = rt
        if q:
            mask = rt.astype(str).apply(lambda c: c.str.contains(q, case=False)).any(axis=1)
            view = rt[mask]
        st.dataframe(view, width="stretch", height=300)
    st.markdown(
        '<div class="note">Only the Low tier is reliable over time (55% of stocks keep their tier when '
        "re-clustered). A log-volatility K-Means gives ordered, more stable tiers (68.8%) but is not adopted in the main tables.</div>",
        unsafe_allow_html=True,
    )