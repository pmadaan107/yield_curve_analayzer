# app.py
# 🇨🇦 Canadian Yield Curve Analyzer — Live via Bank of Canada Valet API
# - Pulls GoC benchmark (par) yields directly from Valet (no uploads/downloads)
# - Compare two specific dates or two rolling windows
# - Charts, slope/curvature diagnostics, plain-English insights

import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Canadian Yield Curve (Live)", layout="wide")
st.title("🇨🇦 Canadian Yield Curve Analyzer (Live)")
st.caption("Government of Canada benchmark (par) yields — Source: Bank of Canada Valet API")

# -------------------------------------------------------------------
# Valet API series codes for benchmark/par GoC yields (percent)
# (These are the correct Valet symbols; do NOT use V12xxxx with Valet.)
# -------------------------------------------------------------------
SERIES: Dict[str, str] = {
    "1Y":  "BD.CDN.1YR.DQ.YLD",
    "2Y":  "BD.CDN.2YR.DQ.YLD",
    "3Y":  "BD.CDN.3YR.DQ.YLD",
    "5Y":  "BD.CDN.5YR.DQ.YLD",
    "7Y":  "BD.CDN.7YR.DQ.YLD",
    "10Y": "BD.CDN.10YR.DQ.YLD",
    "30Y": "BD.CDN.LONG.DQ.YLD",  # long benchmark (~30Y)
}

BASE_URL = "https://www.bankofcanada.ca/valet/observations"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_series(series_codes: List[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Fetch multiple Valet series as a single DataFrame (date index, columns per maturity label).
    start/end format: 'YYYY-MM-DD'
    """
    frames = []
    for code in series_codes:
        url = f"{BASE_URL}/{code}/json"
        params = {}
        if start: params["start_date"] = start
        if end:   params["end_date"] = end
        r = requests.get(url, params=params, timeout=20)
        if not r.ok:
            # Show a short, helpful error; continue so other series can still plot
            st.warning(f"Valet error {r.status_code} for {code}: {r.text[:150]}")
            continue
        j = r.json()
        obs = j.get("observations", [])
        if not obs:
            st.warning(f"No observations returned for {code}.")
            continue
        dates = pd.to_datetime([o["d"] for o in obs])
        vals = [
            float(o[code]["v"]) if o.get(code) and o[code].get("v") not in [None, ""] else np.nan
            for o in obs
        ]
        s = pd.Series(vals, index=dates, name=code)
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).sort_index()
    # Map Valet codes -> human labels (1Y, 2Y, ...)
    return df.rename(columns={v: k for k, v in SERIES.items()})

# -----------------------------
# Pull data (last ~3 years for speed)
# -----------------------------
today = dt.date.today()
start_default = (today - dt.timedelta(days=3*365)).isoformat()
end_default = today.isoformat()

with st.spinner("Fetching Bank of Canada benchmark yields…"):
    df_raw = fetch_series(list(SERIES.values()), start=start_default, end=end_default)

if df_raw.empty:
    st.error("Could not load any benchmark yield data from the Bank of Canada Valet API.")
    st.stop()

# -----------------------------
# Controls
# -----------------------------
st.sidebar.header("🔧 Controls")
mode = st.sidebar.radio("Compare", ["Two specific dates", "Two windows (averages)"], index=0)

min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()

def nearest_row(df: pd.DataFrame, d: dt.date) -> pd.Series:
    idx = df.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
    return df.iloc[idx], df.index[idx].date().isoformat()

if mode == "Two specific dates":
    dA = st.sidebar.date_input("Date A", value=max_date - dt.timedelta(days=30), min_value=min_date, max_value=max_date)
    dB = st.sidebar.date_input("Date B", value=max_date, min_value=min_date, max_value=max_date)
    curveA, labelA = nearest_row(df_raw, dA)
    curveB, labelB = nearest_row(df_raw, dB)
else:
    window_days = st.sidebar.slider("Window length (days)", 5, 90, 30)
    endA = st.sidebar.date_input("Window A end", value=max_date - dt.timedelta(days=30), min_value=min_date, max_value=max_date)
    endB = st.sidebar.date_input("Window B end", value=max_date, min_value=min_date, max_value=max_date)
    startA = (pd.Timestamp(endA) - pd.Timedelta(days=window_days-1)).date()
    startB = (pd.Timestamp(endB) - pd.Timedelta(days=window_days-1)).date()
    winA = df_raw.loc[str(startA):str(endA)]
    winB = df_raw.loc[str(startB):str(endB)]
    curveA = winA.mean(numeric_only=True)
    curveB = winB.mean(numeric_only=True)
    labelA = f"{startA}→{endA}"
    labelB = f"{startB}→{endB}"

# Ensure standard term order for plotting/interp
ordered_terms = [1, 2, 3, 5, 7, 10, 30]
x_terms = []
yA = []
yB = []
for t in ordered_terms:
    lab = f"{t}Y"
    if lab in curveA.index and lab in curveB.index:
        x_terms.append(float(t))
        yA.append(float(curveA[lab]))
        yB.append(float(curveB[lab]))
x_terms = np.array(x_terms, dtype=float)
yA = np.array(yA, dtype=float)
yB = np.array(yB, dtype=float)

if len(x_terms) < 3:
    st.error("Not enough term points to plot a curve. Try different dates/windows.")
    st.stop()

# -----------------------------
# Plots
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_terms, y=yA, mode="lines+markers", name=f"Curve A ({labelA})"))
fig.add_trace(go.Scatter(x=x_terms, y=yB, mode="lines+markers", name=f"Curve B ({labelB})"))
fig.update_layout(
    title="GoC Benchmark Yield Curves (%, par yields)",
    xaxis_title="Maturity (years)",
    yaxis_title="Yield (%)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig, use_container_width=True)

deltas = yB - yA
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=x_terms, y=deltas, name="B − A (pp)"))
fig2.update_layout(
    title="Change in Yields (Curve B − Curve A)",
    xaxis_title="Maturity (years)",
    yaxis_title="Δ Yield (percentage points)",
    template="plotly_white",
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Diagnostics (slope / curvature)
# -----------------------------
def interp(x, y, target):
    return float(np.interp(target, x, y))

sA2, sA10 = interp(x_terms, yA, 2), interp(x_terms, yA, 10)
sB2, sB10 = interp(x_terms, yB, 2), interp(x_terms, yB, 10)

slopeA = sA10 - sA2
slopeB = sB10 - sB2
slope_change = slopeB - slopeA

# curvature proxy using 2Y, 5Y, 10Y
cA = (interp(x_terms, yA, 10) - interp(x_terms, yA, 2)) - (interp(x_terms, yA, 5) - interp(x_terms, yA, 2))
cB = (interp(x_terms, yB, 10) - interp(x_terms, yB, 2)) - (interp(x_terms, yB, 5) - interp(x_terms, yB, 2))
curv_change = cB - cA

st.subheader("📌 Curve Diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("2Y (A → B)", f"{sA2:.2f}% → {sB2:.2f}%", f"{(sB2 - sA2):+.2f} pp")
m2.metric("10Y (A → B)", f"{sA10:.2f}% → {sB10:.2f}%", f"{(sB10 - sA10):+.2f} pp")
m3.metric("Slope (10Y–2Y)", f"{slopeA:.2f}% → {slopeB:.2f}%", f"{slope_change:+.2f} pp")
m4.metric("Curvature Δ", f"{cA:.2f} → {cB:.2f}", f"{curv_change:+.2f}")

# -----------------------------
# Auto-Insights (plain English)
# -----------------------------
def insights_text(slope_change, sA2, sB2, sA10, sB10):
    msgs = []
    # Slope narrative
    if slope_change > 0.10:
        msgs.append("**Steepening:** Long yields rose vs short; markets may be pricing stronger growth/inflation or slower policy easing.")
    elif slope_change < -0.10:
        msgs.append("**Flattening/Inversion:** Short yields rose vs long (or fell less), pointing to tighter near-term policy or growth concerns.")
    else:
        msgs.append("**Little slope change:** Overall shape broadly unchanged.")

    # Directional move
    if (sB2 - sA2) > 0.10 and (sB10 - sA10) > 0:
        msgs.append("**Broad shift up:** Funding costs higher across maturities.")
    elif (sB2 - sA2) < -0.10 and (sB10 - sA10) < 0:
        msgs.append("**Broad shift down:** Borrowing conditions may ease if this persists.")
    else:
        msgs.append("**Mixed moves:** Short and long ends moved differently; watch credit spreads and issuance windows.")

    # Practical implications
    if slope_change > 0:
        msgs.append("- **Borrowers:** Consider terming out debt before long-end yields climb further.")
        msgs.append("- **Investors:** Steepener views (long 10Y vs short 2Y) may benefit if the trend continues.")
    else:
        msgs.append("- **Borrowers:** Short-dated funding may remain costly; hedge near-term cash flows.")
        msgs.append("- **Investors:** Flattener views can benefit; be selective with rate-sensitive sectors.")
    return "\n".join([f"- {m}" for m in msgs])

st.subheader("🧠 Auto-Insights")
st.markdown(insights_text(slope_change, sA2, sB2, sA10, sB10))

st.caption(
    "Data: Bank of Canada Valet API (benchmark/par Government of Canada yields). "
    "App auto-fetches live series and caches for 1 hour."
)


