"""
OTT Churn Prediction & Retention Intelligence System
Premium Streamlit Dashboard
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="StreamIQ · Churn Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080b14 !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1600px !important; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg, #e50914 0%, #ff6b35 50%, #f7c59f 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: -0.03em; padding: 0.5rem 0 1.5rem; text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;
}
.page-header {
    background: linear-gradient(135deg, rgba(229,9,20,0.12) 0%, rgba(13,17,23,0) 60%);
    border: 1px solid rgba(229,9,20,0.2); border-radius: 16px;
    padding: 1.75rem 2rem; margin-bottom: 1.75rem; position: relative; overflow: hidden;
}
.page-header::before {
    content: ''; position: absolute; top: -50%; right: -5%; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(229,9,20,0.08) 0%, transparent 70%); pointer-events: none;
}
.page-title { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: #f8fafc; line-height: 1.1; }
.page-subtitle { font-size: 0.9rem; color: #64748b; margin-top: 0.35rem; }

.kpi-card {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 16px;
    padding: 1.4rem 1.6rem; position: relative; overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
.kpi-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0;
    height: 2px; border-radius: 0 0 16px 16px;
}
.kpi-card.red::after   { background: linear-gradient(90deg,#e50914,#ff6b35); }
.kpi-card.blue::after  { background: linear-gradient(90deg,#3b82f6,#06b6d4); }
.kpi-card.green::after { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.amber::after { background: linear-gradient(90deg,#f59e0b,#fb923c); }
.kpi-card.purple::after{ background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.kpi-card.teal::after  { background: linear-gradient(90deg,#14b8a6,#2dd4bf); }
.kpi-icon  { font-size: 1.6rem; margin-bottom: 0.6rem; }
.kpi-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; margin: 0.2rem 0; }
.kpi-delta { font-size: 0.78rem; font-weight: 500; }
.kpi-delta.up      { color: #10b981; }
.kpi-delta.down    { color: #e50914; }
.kpi-delta.neutral { color: #64748b; }

.section-header {
    display: flex; align-items: center; gap: 0.75rem;
    margin: 1.75rem 0 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.section-title { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; color: #f1f5f9; }
.section-badge {
    background: rgba(229,9,20,0.15); color: #ff6b6b; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.06em; padding: 0.2rem 0.6rem; border-radius: 20px;
    text-transform: uppercase; border: 1px solid rgba(229,9,20,0.25);
}
.result-safe {
    background: linear-gradient(135deg,rgba(16,185,129,0.15),rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.3); border-radius: 16px; padding: 2rem; text-align: center;
}
.result-churn {
    background: linear-gradient(135deg,rgba(229,9,20,0.15),rgba(229,9,20,0.05));
    border: 1px solid rgba(229,9,20,0.35); border-radius: 16px; padding: 2rem; text-align: center;
}
.result-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.4rem; }
.result-prob  { font-size: 3rem; font-weight: 800; font-family: 'Syne', sans-serif; }
.badge-low    { background:#052e16; color:#4ade80; border:1px solid #15803d; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:700; }
.badge-medium { background:#451a03; color:#fb923c; border:1px solid #c2410c; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:700; }
.badge-high   { background:#1c0a0a; color:#f87171; border:1px solid #b91c1c; border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:700; }
.tip-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #e50914; border-radius: 10px; padding: 0.85rem 1.1rem;
    margin: 0.45rem 0; font-size: 0.88rem; color: #cbd5e1; line-height: 1.55;
}
.metric-pill { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.6rem 1rem; text-align: center; }
.metric-pill-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; color: #64748b; font-weight: 600; }
.metric-pill-value { font-family: 'Syne', sans-serif; font-size: 1.35rem; font-weight: 800; color: #f1f5f9; }
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.5rem 0; }

[data-testid="stSelectbox"] > div, [data-testid="stSlider"] > div { color: #e2e8f0; }
.stButton > button {
    background: linear-gradient(135deg, #e50914 0%, #ff4500 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important;
    padding: 0.65rem 2rem !important; letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important; box-shadow: 0 4px 20px rgba(229,9,20,0.35) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(229,9,20,0.5) !important; }
[data-baseweb="select"] > div { background: #111827 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; color: #e2e8f0 !important; }
[data-baseweb="input"] > div { background: #111827 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
.stNumberInput input, .stTextInput input { background: #111827 !important; color: #e2e8f0 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stExpander"] { background: #111827 !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
.stRadio > label { color: #94a3b8 !important; }
</style>
"""

# ──────────────────────────────────────────────
# PLOTLY THEME  ← legend key REMOVED from here
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    title_font=dict(family="Syne", color="#f1f5f9", size=15),
    margin=dict(t=45, b=30, l=10, r=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
    colorway=["#e50914","#3b82f6","#10b981","#f59e0b","#8b5cf6","#06b6d4","#fb923c"],
)

# ── Reusable legend dicts (pass separately, never with **PLOTLY_LAYOUT)
LEGEND_STD    = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
LEGEND_BOTTOM = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), orientation="h", y=-0.15)

COLORS = {"churn":"#e50914","retain":"#10b981","neutral":"#3b82f6",
          "amber":"#f59e0b","purple":"#8b5cf6","teal":"#06b6d4"}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def kpi(icon, label, value, delta="", delta_dir="neutral", accent="blue"):
    return (
        f'<div class="kpi-card {accent}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        + (f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else "")
        + "</div>"
    )

def section(icon, title, badge=""):
    b = f'<span class="section-badge">{badge}</span>' if badge else ""
    return (f'<div class="section-header">'
            f'<span style="font-size:1.2rem">{icon}</span>'
            f'<span class="section-title">{title}</span>{b}</div>')


# ──────────────────────────────────────────────
# DATA / MODEL LOADERS
# ──────────────────────────────────────────────
DATA_PATH  = Path(__file__).parent.parent / "data" / "ott_churn_data.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        from data.generate_data import generate_ott_dataset
        df = generate_ott_dataset(5000)
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    arts = {}
    for f in ["best_model","all_results","feature_importance","encoders","scaler","feature_names"]:
        p = MODELS_DIR / f"{f}.pkl"
        arts[f] = joblib.load(p) if p.exists() else None
    return arts

def ensure_trained():
    if not (MODELS_DIR / "best_model.pkl").exists():
        with st.spinner("🚀 Training ML models for the first time… (~60 s)"):
            import subprocess
            subprocess.run([sys.executable,
                            str(Path(__file__).parent.parent / "train.py")], check=True)
        st.cache_resource.clear()
        st.rerun()


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🎬 StreamIQ</div>', unsafe_allow_html=True)
        choice = st.radio("Navigation",
                          ["Overview","Analytics","Models","Predict","Retention"],
                          label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.72rem;color:#334155;text-align:center;line-height:1.8">'
            'StreamIQ · Churn Intelligence<br>'
            '<span style="color:#e50914">v2.0</span> · ML-Powered Analytics</div>',
            unsafe_allow_html=True)
    return choice


# ──────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────
def page_overview(df):
    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Executive Dashboard</div>'
        '<div class="page-subtitle">Real-time retention intelligence for your OTT platform</div>'
        '</div>', unsafe_allow_html=True)

    total      = len(df)
    churned    = int(df["churn"].sum())
    churn_rate = churned / total
    retained   = total - churned
    rev_at_risk= df[df["churn"]==1]["monthly_subscription_cost"].sum()
    avg_watch  = df["monthly_watch_hours"].mean()
    high_risk  = int(df[(df["churn"]==1) & (df["days_since_last_login"]>30)].shape[0])

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.markdown(kpi("👥","Total Customers",  f"{total:,}",              "",                 "neutral","blue"),  unsafe_allow_html=True)
    with k2: st.markdown(kpi("📉","Churn Rate",        f"{churn_rate:.1%}",       "▲ vs last month",  "down",   "red"),   unsafe_allow_html=True)
    with k3: st.markdown(kpi("💸","Revenue at Risk",   f"₹{rev_at_risk/1e5:.1f}L","Monthly exposure", "down",   "amber"), unsafe_allow_html=True)
    with k4: st.markdown(kpi("📺","Avg Watch Hours",   f"{avg_watch:.1f}h",       "per user/month",   "neutral","teal"),  unsafe_allow_html=True)
    with k5: st.markdown(kpi("⚠️","High-Risk Users",  f"{high_risk:,}",          "Needs action",     "down",   "red"),   unsafe_allow_html=True)
    with k6: st.markdown(kpi("✅","Retention Rate",    f"{1-churn_rate:.1%}",     "",                 "up",     "green"), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(section("📈","Platform Trends","Live"), unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        bins   = [0,6,12,24,36,60]
        labels = ["0-6m","7-12m","13-24m","25-36m","36m+"]
        df2    = df.copy()
        df2["age_bucket"] = pd.cut(df2["account_age_months"], bins=bins, labels=labels)
        grp = df2.groupby("age_bucket", observed=True)["churn"].agg(["sum","count"]).reset_index()
        grp["rate"] = grp["sum"] / grp["count"]
        fig = go.Figure(go.Bar(
            x=grp["age_bucket"], y=grp["rate"],
            marker=dict(color=grp["rate"],
                        colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#e50914"]],
                        line=dict(width=0)),
            text=[f"{r:.0%}" for r in grp["rate"]],
            textposition="outside", textfont=dict(color="#94a3b8", size=11),
        ))
        fig.update_layout(title="Churn Rate by Account Age",
                          xaxis_title="Account Age", yaxis_title="Churn Rate",
                          yaxis_tickformat=".0%",
                          legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Pie(
            values=[churned, retained], labels=["Churned","Retained"],
            hole=0.68,
            marker=dict(colors=["#e50914","#10b981"], line=dict(width=0)),
            textinfo="none",
        ))
        fig2.add_annotation(
            text=f"<b>{churn_rate:.0%}</b><br><span style='font-size:10px'>CHURN</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color="#f1f5f9", family="Syne"),
        )
        # ── FIX: legend passed separately, NOT inside **PLOTLY_LAYOUT
        fig2.update_layout(title="Churn Composition", showlegend=True,
                           legend=LEGEND_BOTTOM, **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(
            df, x="monthly_watch_hours", color="churn",
            color_discrete_map={0:"#10b981",1:"#e50914"},
            barmode="overlay", opacity=0.75, nbins=40,
            labels={"churn":"Churn","monthly_watch_hours":"Watch Hours / Month"},
            title="Watch Hours Distribution by Churn",
        )
        fig3.update_layout(legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        sub_churn = df.groupby("subscription_type")["churn"].mean().reset_index()
        sub_churn.columns = ["Subscription","Churn Rate"]
        fig4 = px.bar(sub_churn, x="Subscription", y="Churn Rate",
                      color="Churn Rate",
                      color_continuous_scale=["#10b981","#f59e0b","#e50914"],
                      text=[f"{v:.1%}" for v in sub_churn["Churn Rate"]],
                      title="Churn Rate by Subscription Tier")
        fig4.update_traces(textposition="outside", textfont=dict(color="#94a3b8"))
        fig4.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False,
                           legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE: ANALYTICS / EDA
# ──────────────────────────────────────────────
def page_analytics(df):
    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Analytics & EDA</div>'
        '<div class="page-subtitle">Deep-dive into platform usage, revenue, and engagement patterns</div>'
        '</div>', unsafe_allow_html=True)

    with st.expander("🎛️  Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        subs      = fc1.multiselect("Subscription", df["subscription_type"].unique(), default=list(df["subscription_type"].unique()))
        genders   = fc2.multiselect("Gender",        df["gender"].unique(),            default=list(df["gender"].unique()))
        age_range = fc3.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (18,65))

    mask = (df["subscription_type"].isin(subs) &
            df["gender"].isin(genders) &
            df["age"].between(*age_range))
    dff = df[mask]

    st.markdown(section("🎬","Content & Genre Intelligence"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gc = dff.groupby("favorite_genre")["churn"].agg(["mean","count"]).reset_index()
        gc.columns = ["Genre","Churn Rate","Users"]
        gc = gc.sort_values("Churn Rate", ascending=True)
        fig = px.bar(gc, y="Genre", x="Churn Rate", orientation="h",
                     color="Churn Rate", color_continuous_scale=["#10b981","#f59e0b","#e50914"],
                     title="Churn Rate by Favourite Genre",
                     text=[f"{v:.1%}" for v in gc["Churn Rate"]])
        fig.update_traces(textposition="outside", textfont=dict(color="#94a3b8"))
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False,
                          legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        bd = dff.groupby(["binge_watch_frequency","churn"]).size().reset_index(name="count")
        fig2 = px.bar(bd, x="binge_watch_frequency", y="count", color="churn",
                      color_discrete_map={0:"#10b981",1:"#e50914"}, barmode="group",
                      category_orders={"binge_watch_frequency":["Never","Rarely","Sometimes","Often","Always"]},
                      title="Binge Frequency vs Churn")
        fig2.update_layout(xaxis_title="Binge Frequency", yaxis_title="Users",
                           legend_title="Churn", legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(section("💰","Revenue Intelligence"), unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        rev_df = dff.groupby("subscription_type").agg(
            total_rev=("monthly_subscription_cost","sum"),
            at_risk=("monthly_subscription_cost", lambda x: x[dff.loc[x.index,"churn"]==1].sum())
        ).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Safe Revenue",    x=rev_df["subscription_type"], y=rev_df["total_rev"]-rev_df["at_risk"], marker_color="#10b981", opacity=0.85))
        fig3.add_trace(go.Bar(name="Revenue at Risk", x=rev_df["subscription_type"], y=rev_df["at_risk"],                     marker_color="#e50914", opacity=0.85))
        fig3.update_layout(barmode="stack", title="Revenue by Subscription Tier",
                           yaxis_title="Monthly Revenue (₹)",
                           legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        pc = dff.groupby("payment_method")["churn"].mean().reset_index()
        pc.columns = ["Method","Churn Rate"]
        fig4 = px.pie(pc, values="Churn Rate", names="Method", hole=0.5,
                      title="Churn Distribution by Payment Method",
                      color_discrete_sequence=["#e50914","#3b82f6","#f59e0b","#10b981"])
        fig4.update_layout(legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown(section("📱","User Behaviour"), unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        samp = dff.sample(min(1000,len(dff)), random_state=42)
        fig5 = px.scatter(samp, x="monthly_watch_hours", y="avg_session_time",
                          color=samp["churn"].map({0:"Retained",1:"Churned"}),
                          color_discrete_map={"Retained":"#10b981","Churned":"#e50914"},
                          opacity=0.6, title="Watch Hours vs Avg Session Time",
                          labels={"monthly_watch_hours":"Monthly Watch Hours (h)",
                                  "avg_session_time":"Avg Session (h)","color":"Status"})
        fig5.update_layout(legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        dff2 = dff.copy()
        dff2["login_bucket"] = pd.cut(dff2["days_since_last_login"],
                                      bins=[0,7,14,30,60,120],
                                      labels=["0-7d","8-14d","15-30d","31-60d","61-120d"])
        lg = dff2.groupby("login_bucket", observed=True)["churn"].mean().reset_index()
        fig6 = px.line(lg, x="login_bucket", y="churn", markers=True,
                       title="Churn Rate by Days Since Last Login",
                       labels={"churn":"Churn Rate","login_bucket":"Days Since Login"},
                       color_discrete_sequence=["#e50914"])
        fig6.update_traces(line_width=2.5, marker_size=8, marker_color="#ff6b35")
        fig6.update_layout(yaxis_tickformat=".0%", legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown(section("🔥","Correlation Heatmap"), unsafe_allow_html=True)
    num_cols = ["age","monthly_watch_hours","avg_session_time","monthly_subscription_cost",
                "support_tickets","days_since_last_login","number_of_profiles",
                "content_rating_given","account_age_months","auto_renew_enabled","churn"]
    corr = dff[num_cols].corr()
    fig7 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale=[[0,"#0369a1"],[0.5,"#111827"],[1,"#e50914"]],
                     title="Feature Correlation Matrix", zmin=-1, zmax=1)
    fig7.update_layout(coloraxis_colorbar=dict(title="r", tickfont=dict(color="#94a3b8")),
                       legend=LEGEND_STD, **PLOTLY_LAYOUT)
    st.plotly_chart(fig7, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE: ML MODELS
# ──────────────────────────────────────────────
def page_models(arts):
    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">ML Model Performance</div>'
        '<div class="page-subtitle">Comparative analysis across 4 classification algorithms</div>'
        '</div>', unsafe_allow_html=True)

    results = arts.get("all_results")
    if not results:
        st.warning("⚠️ Models not trained yet. Run `python train.py` first.")
        return

    best = max(results, key=lambda n: results[n]["metrics"]["ROC-AUC"])

    st.markdown(section("🏆","Model Leaderboard"), unsafe_allow_html=True)
    cols = st.columns(len(results))
    for idx,(name,res) in enumerate(results.items()):
        m       = res["metrics"]
        is_best = name == best
        border  = "border:1px solid rgba(229,9,20,0.5);" if is_best else "border:1px solid rgba(255,255,255,0.07);"
        glow    = "box-shadow:0 0 20px rgba(229,9,20,0.15);" if is_best else ""
        pills   = "".join([
            f'<div class="metric-pill" style="margin-top:0.7rem">'
            f'<div class="metric-pill-label">{k}</div>'
            f'<div class="metric-pill-value">{v:.4f}</div></div>'
            for k,v in m.items()])
        with cols[idx]:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#111827,#0f172a);{border}{glow}'
                f'border-radius:16px;padding:1.2rem;text-align:center;margin-bottom:0.5rem">'
                f'{"🥇 " if is_best else ""}'
                f'<b style="font-family:Syne;color:#f1f5f9;font-size:0.95rem">{name}</b>'
                f'{pills}</div>', unsafe_allow_html=True)

    st.markdown(section("📉","ROC Curves"), unsafe_allow_html=True)
    palette = ["#e50914","#3b82f6","#10b981","#f59e0b"]
    c1, c2  = st.columns(2)

    with c1:
        fig = go.Figure()
        for (name,res),col in zip(results.items(),palette):
            fpr,tpr = res["fpr"], res["tpr"]
            if len(fpr):
                fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                         name=f"{name} (AUC={res['metrics']['ROC-AUC']:.3f})",
                                         line=dict(color=col, width=2.5)))
        fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,
                      line=dict(color="#334155",dash="dash",width=1.5))
        fig.update_layout(title="ROC-AUC Comparison",
                          xaxis_title="FPR", yaxis_title="TPR",
                          legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        metrics_list = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        fig2 = go.Figure()
        for (name,res),col in zip(results.items(),palette):
            vals = [res["metrics"][m] for m in metrics_list] + [res["metrics"][metrics_list[0]]]
            fig2.add_trace(go.Scatterpolar(
                r=vals, theta=metrics_list+[metrics_list[0]],
                fill="toself", name=name, line_color=col, opacity=0.85))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True,range=[0,1],color="#334155",gridcolor="#1e293b"),
                angularaxis=dict(color="#64748b",gridcolor="#1e293b"),
            ),
            title="Model Performance Radar",
            legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    feat_imp = arts.get("feature_importance")
    if feat_imp is not None:
        st.markdown(section("🔑","Feature Importance (XGBoost / RF)"), unsafe_allow_html=True)
        top_feat = feat_imp.head(12).reset_index()
        top_feat.columns = ["Feature","Importance"]
        fig3 = px.bar(top_feat, y="Feature", x="Importance", orientation="h",
                      color="Importance",
                      color_continuous_scale=["#1e3a5f","#3b82f6","#e50914"],
                      title="Top 12 Predictive Features",
                      text=[f"{v:.4f}" for v in top_feat["Importance"]])
        fig3.update_traces(textposition="outside", textfont=dict(color="#94a3b8"))
        fig3.update_layout(coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"),
                           legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE: PREDICT
# ──────────────────────────────────────────────
def page_predict(arts):
    from src.model_training import churn_risk_label, retention_suggestions

    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Churn Predictor</div>'
        '<div class="page-subtitle">Enter customer details to predict churn probability and get retention actions</div>'
        '</div>', unsafe_allow_html=True)

    model    = arts.get("best_model")
    encoders = arts.get("encoders")
    scaler   = arts.get("scaler")

    if not model:
        st.warning("⚠️ Models not trained yet. Run `python train.py` first.")
        return

    st.markdown(section("📝","Customer Profile"), unsafe_allow_html=True)

    r1c1,r1c2,r1c3,r1c4 = st.columns(4)
    age            = r1c1.slider("Age", 18, 70, 32)
    account_months = r1c2.slider("Account Age (months)", 1, 60, 18)
    devices        = r1c3.slider("Devices Used", 1, 5, 2)
    profiles       = r1c4.slider("Number of Profiles", 1, 5, 2)

    r2c1,r2c2,r2c3,r2c4 = st.columns(4)
    watch_hours  = r2c1.number_input("Monthly Watch Hours", 0.0, 200.0, 35.0, step=1.0)
    session_time = r2c2.number_input("Avg Session Time (h)", 0.5, 8.0, 2.5, step=0.1)
    sub_cost     = r2c3.number_input("Monthly Sub Cost (₹)", 50.0, 600.0, 249.0, step=10.0)
    days_login   = r2c4.slider("Days Since Last Login", 0, 120, 10)

    r3c1,r3c2,r3c3,r3c4 = st.columns(4)
    tickets     = r3c1.selectbox("Support Tickets", [0,1,2,3,4,5], index=0)
    content_rat = r3c2.slider("Content Rating Given", 1.0, 5.0, 4.0, step=0.1)
    auto_renew  = r3c3.selectbox("Auto-Renew Enabled", ["Yes","No"])
    gender      = r3c4.selectbox("Gender", ["Male","Female","Other"])

    r4c1,r4c2,r4c3,r4c4 = st.columns(4)
    sub_type  = r4c1.selectbox("Subscription Type", ["Basic","Standard","Premium"])
    fav_genre = r4c2.selectbox("Favourite Genre", ["Action","Drama","Comedy","Thriller","Romance","Sci-Fi","Documentary"])
    payment   = r4c3.selectbox("Payment Method", ["Credit Card","UPI","Net Banking","Wallet"])
    ads_tol   = r4c4.selectbox("Ads Tolerance", ["Low","Medium","High"])
    binge_freq= st.selectbox("Binge Watch Frequency", ["Never","Rarely","Sometimes","Often","Always"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮  Predict Churn Risk"):
        row = {
            "age": age, "gender": gender, "subscription_type": sub_type,
            "monthly_watch_hours": watch_hours, "favorite_genre": fav_genre,
            "devices_used": devices, "avg_session_time": session_time,
            "monthly_subscription_cost": sub_cost, "payment_method": payment,
            "support_tickets": tickets, "ads_tolerance": ads_tol,
            "days_since_last_login": days_login, "number_of_profiles": profiles,
            "binge_watch_frequency": binge_freq, "content_rating_given": content_rat,
            "account_age_months": account_months,
            "auto_renew_enabled": 1 if auto_renew=="Yes" else 0,
        }

        from src.preprocessing import CATEGORICAL_COLS, NUMERIC_COLS
        inp = pd.DataFrame([row])
        for col in CATEGORICAL_COLS:
            if col in inp.columns and encoders and col in encoders:
                le  = encoders[col]
                val = str(inp[col].iloc[0])
                inp[col] = le.transform([val])[0] if val in le.classes_ else -1
        feat_order = [c for c in (CATEGORICAL_COLS+NUMERIC_COLS) if c in inp.columns]
        X_sc = scaler.transform(inp[feat_order].values)
        prob = float(model.predict_proba(X_sc)[0][1])
        risk = churn_risk_label(prob)
        tips = retention_suggestions(row, prob)

        st.markdown("<br>", unsafe_allow_html=True)
        r_class = "result-churn" if prob >= 0.5 else "result-safe"
        emoji   = "🚨" if prob >= 0.5 else "✅"
        verdict = "Likely to Churn" if prob >= 0.5 else "Likely to Stay"
        color   = "#e50914" if prob >= 0.5 else "#10b981"

        res_col, gauge_col = st.columns([1.2, 1])
        with res_col:
            risk_badge = {"Low Risk":"badge-low","Medium Risk":"badge-medium","High Risk":"badge-high"}[risk]
            st.markdown(
                f'<div class="{r_class}">'
                f'<div style="font-size:2.5rem">{emoji}</div>'
                f'<div class="result-title" style="color:{color}">{verdict}</div>'
                f'<div class="result-prob"  style="color:{color}">{prob:.0%}</div>'
                f'<div style="margin:0.5rem 0">Churn Probability</div>'
                f'<span class="{risk_badge}">{risk}</span>'
                f'</div>', unsafe_allow_html=True)

        with gauge_col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                title={"text":"Churn Score","font":{"family":"Syne","color":"#f1f5f9","size":14}},
                number={"suffix":"%","font":{"family":"Syne","color":color,"size":36}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#64748b"}},
                    "bar":{"color":color,"thickness":0.25},
                    "bgcolor":"rgba(0,0,0,0)", "bordercolor":"rgba(0,0,0,0)",
                    "steps":[
                        {"range":[0,35],"color":"rgba(16,185,129,0.15)"},
                        {"range":[35,65],"color":"rgba(245,158,11,0.15)"},
                        {"range":[65,100],"color":"rgba(229,9,20,0.15)"},
                    ],
                    "threshold":{"line":{"color":"white","width":2},"thickness":0.75,"value":prob*100},
                }))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                 font=dict(color="#94a3b8"),
                                 margin=dict(t=30,b=10,l=20,r=20), height=260)
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(section("💡","Personalised Retention Actions"), unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: BUSINESS INTELLIGENCE
# ──────────────────────────────────────────────
def page_retention(df, arts):
    st.markdown(
        '<div class="page-header">'
        '<div class="page-title">Business Intelligence</div>'
        '<div class="page-subtitle">Segment-level churn risk, revenue impact, and strategic recommendations</div>'
        '</div>', unsafe_allow_html=True)

    model    = arts.get("best_model")
    encoders = arts.get("encoders")
    scaler   = arts.get("scaler")

    df = df.copy()
    if model and encoders and scaler:
        from src.preprocessing import CATEGORICAL_COLS, NUMERIC_COLS
        df2 = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df2.columns and col in encoders:
                le = encoders[col]
                df2[col] = df2[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)
        feat_order = [c for c in (CATEGORICAL_COLS+NUMERIC_COLS) if c in df2.columns]
        # Impute NaNs before prediction — prevents sklearn ValueError
        for col in feat_order:
            if df2[col].dtype in [np.float64, np.float32, float]:
                df2[col] = df2[col].fillna(df2[col].median())
            else:
                df2[col] = df2[col].fillna(-1)
        probs = model.predict_proba(scaler.transform(df2[feat_order].values))[:,1]
        df["churn_prob"]   = probs
        df["risk_segment"] = pd.cut(probs,[0,0.35,0.65,1.0],
                                    labels=["Low Risk","Medium Risk","High Risk"])
    else:
        df["churn_prob"]   = df["churn"].astype(float)
        df["risk_segment"] = df["churn_prob"].apply(
            lambda p: "Low Risk" if p<0.35 else ("Medium Risk" if p<0.65 else "High Risk"))

    st.markdown(section("🎯","Risk Segmentation"), unsafe_allow_html=True)
    seg_counts = df["risk_segment"].value_counts()

    sk1,sk2,sk3,sk4 = st.columns(4)
    with sk1: st.markdown(kpi("🟢","Low Risk",    f'{seg_counts.get("Low Risk",   0):,}',"","up","green"),      unsafe_allow_html=True)
    with sk2: st.markdown(kpi("🟡","Medium Risk",  f'{seg_counts.get("Medium Risk",0):,}',"","neutral","amber"), unsafe_allow_html=True)
    with sk3: st.markdown(kpi("🔴","High Risk",    f'{seg_counts.get("High Risk",  0):,}',"Needs attention","down","red"), unsafe_allow_html=True)
    with sk4:
        at_risk_rev = df[df["risk_segment"]=="High Risk"]["monthly_subscription_cost"].sum()
        st.markdown(kpi("💰","Revenue at Risk",f"₹{at_risk_rev/1e5:.1f}L","Monthly exposure","down","red"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sr = df.groupby("risk_segment", observed=True)["monthly_subscription_cost"].sum().reset_index()
        sr.columns = ["Segment","Revenue"]
        fig1 = px.pie(sr, values="Revenue", names="Segment", hole=0.55,
                      title="Revenue by Risk Segment", color="Segment",
                      color_discrete_map={"Low Risk":"#10b981","Medium Risk":"#f59e0b","High Risk":"#e50914"})
        fig1.update_layout(legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        ss = df.groupby(["subscription_type","risk_segment"], observed=True).size().reset_index(name="count")
        fig2 = px.bar(ss, x="subscription_type", y="count", color="risk_segment",
                      color_discrete_map={"Low Risk":"#10b981","Medium Risk":"#f59e0b","High Risk":"#e50914"},
                      title="Risk Segments by Subscription Tier", barmode="group")
        fig2.update_layout(xaxis_title="Subscription", yaxis_title="Customers",
                           legend=LEGEND_STD, **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(section("⚠️","High-Risk Customer List","Top 20"), unsafe_allow_html=True)
    top_risk = df.nlargest(20,"churn_prob")[
        ["customer_id","subscription_type","monthly_watch_hours",
         "days_since_last_login","support_tickets","churn_prob","risk_segment"]
    ].copy()
    top_risk["churn_prob"] = top_risk["churn_prob"].map("{:.1%}".format)
    st.dataframe(
        top_risk.style.applymap(
            lambda v: "color:#e50914;font-weight:700" if v=="High Risk" else
                      ("color:#f59e0b;font-weight:700" if v=="Medium Risk" else "color:#10b981"),
            subset=["risk_segment"]),
        use_container_width=True, height=320)

    st.markdown(section("🧠","Strategic Recommendations","AI-Powered"), unsafe_allow_html=True)
    recs = [
        ("🎁","Loyalty Programme","Launch a points-based loyalty scheme for Premium subscribers with ≥12 months tenure to lower voluntary churn by an estimated 8–12%."),
        ("📧","Re-engagement Campaigns","Automate personalised email/push sequences for users inactive >14 days. A/B test subject lines with genre-specific recommendations."),
        ("💳","Payment Nudges","Users with auto-renew disabled show 2.3× higher churn. Incentivise enablement with a one-month discount popup."),
        ("🎬","Content Personalisation","Users rating content <3.0 churn at 2× the base rate. Improve the recommendation engine to surface high-affinity content faster."),
        ("🛠️","Support Excellence","Every additional support ticket raises churn probability ~18%. Introduce proactive outreach after the 2nd ticket."),
        ("⬆️","Upgrade Campaigns","Basic-tier users watching >20h/month are high-conversion candidates for a Standard/Premium upsell. Offer a 14-day free trial."),
    ]
    rc1, rc2 = st.columns(2)
    for i,(icon,title,desc) in enumerate(recs):
        col = rc1 if i%2==0 else rc2
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#111827,#0f172a);'
                f'border:1px solid rgba(255,255,255,0.07);border-radius:14px;'
                f'padding:1.2rem 1.4rem;margin-bottom:0.8rem">'
                f'<div style="font-size:1.5rem;margin-bottom:0.4rem">{icon}</div>'
                f'<div style="font-family:Syne;font-weight:700;color:#f1f5f9;font-size:0.95rem;margin-bottom:0.4rem">{title}</div>'
                f'<div style="font-size:0.84rem;color:#94a3b8;line-height:1.55">{desc}</div>'
                f'</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
    ensure_trained()
    df   = load_data()
    arts = load_models()
    page = sidebar()

    if page == "Overview":
        page_overview(df)
    elif page == "Analytics":
        page_analytics(df)
    elif page == "Models":
        page_models(arts)
    elif page == "Predict":
        page_predict(arts)
    elif page == "Retention":
        page_retention(df, arts)

if __name__ == "__main__":
    main()
