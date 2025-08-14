# app.py
# 🇨🇦 Canadian Yield Curve Analyzer (Live via Bank of Canada Valet API)
# - Directly fetches Government of Canada benchmark bond yields (par yields)
# - Compare two dates or two rolling windows
# - Detect steepening/flattening and provide business/investor insights
# Sources: BoC Selected Bond Yields (Valet API) + docs

import datetime as dt
import io
import json
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Canadian Yield Curve (Live)", layout="wide")
st.title("🇨🇦 Canadian Yield Curve Analyzer (Live)")
st.caption("Government of Canada benchmark (par) yields via Bank of Canada Valet API")

# -------------------------------------------------------------------
# CONFIG — BoC Valet API series for GoC benchmark/par yields (percent)
# You can expand this dict if you want more points on the curve.
# Keys = label to show; Values = Valet series code.
# Common benchmarks exposed via Valet include 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, Long (e.g., 30Y).
#
# Tip: If you want even denser points, you can also use 'marketable bond average yields' series.
# -------------------------------------------------------------------
SERIES: Dict[str, str] = {
    "1Y":  "V122544",  # 1-year GoC benchmark yield (%)
    "2Y":  "V122545",
    "3Y":  "V122546",
    "5Y":  "V122547",
    "7Y":  "V122548",
    "10Y": "V122550",
    "30Y": "V122551",  # Long/30-year benchmark
}

BASE_URL = "https://www.bankofcanada.ca/valet/observations"

@st.cache_data(ttl=3600)
def fetch_series(series_codes: List[str], start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch multiple Valet series as a single DataFrame (date index, columns per series).
    start/end format: 'YYYY-MM-DD'
    """
    cols = []
    frames = []
    for code in series_codes:
        url = f"{BASE_URL}/{code}/json"
        params = {}
        if start:
            params["start_date"] = start
        if end:
            params["end_date"] = end
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        obs = j.get("observations", [])
        dates = [o["d"] for o in obs]
        vals = [float(o[code]["v"]) if o.get(code, {}).get("v") not in [None, ""] else np.nan for o in obs]
        s = pd.Series(vals, index=pd.to_datetime(dates), name=code)
        frames.append(s)
        cols.append(code)
    df = pd.concat(frames, axis=1).sort_index()
    # Rename to maturity labels
    mapper = {v: k for k, v in SERIES.items()}
    df = df.rename(columns=mapper)
    return df

# -----------------------------
# Pull data (last 3 years by default for speed)
# -----------------------------
today = dt.date.today()
start_default = (today - dt.timedelta(days=3*365)).isoformat()
end_default = today.isoformat()

with st.spinner("Fetching Bank of Canada benchmark yields…"):
    df_raw = fetch_series(list(SERIES.values()), start=start_default, end=end_default)

if df_raw.empty:
    st.error("Could not load data from the Bank of Canada Valet API. Please try again later.")
    st.stop()

# -----------------------------
# Controls
# -----------------------------
st.sidebar.header("🔧 Controls")
mode = st.sidebar.radio("Compare", ["Two specific dates", "Two windows (averages)"], index=0)

min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()

if mode == "Two specific dates":
    dA = st.sidebar.date_input("Date A", value=max_date - dt.timedelta(days=30), min_value=min_date, max_value=max_date)
    dB = st.sidebar.date_input("Date B", value=max_date, min_value=min_date, max_value=max_date)
    idxA = df_raw.index.get_indexer([pd.Timestamp(dA)], method="nearest")[0]
    idxB = df_raw.index.get_indexer([pd.Timestamp(dB)], method="nearest")[0]
    curveA = df_raw.iloc[idxA]
    curveB = df_raw.iloc[idxB]
    labelA = df_raw.index[idxA].date().isoformat()
    labelB = df_raw.index[idxB].date().isoformat()
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

# Ensure term order (1,2,3,5,7,10,30)
terms_years = []
valsA = []
valsB = []
for k in ["1Y","2Y","3Y","5Y","7Y","10Y","30Y"]:
    if k in curveA.index and k in curveB.index:
        terms_years.append(int(k.replace("Y","")))
        valsA.append(curveA[k])
        valsB.append(curveB[k])
terms_years = np.array(terms_years, dtype=float)
valsA = np.array(valsA, dtype=float)
valsB = np.array(valsB, dtype=float)
deltas = valsB - valsA

# -----------------------------
# Plots
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=terms_years, y=valsA, mode="lines+markers", name=f"Curve A ({labelA})"))
fig.add_trace(go.Scatter(x=terms_years, y=valsB, mode="lines+markers", name=f"Curve B ({labelB})"))
fig.update_layout(
    title="GoC Benchmark Yield Curves (%, par yields)",
    xaxis_title="Maturity (years)",
    yaxis_title="Yield (%)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=terms_years, y=deltas, name="B − A (pp)"))
fig2.update_layout(
    title="Change in Yields (Curve B − Curve A)",
    xaxis_title="Maturity (years)",
    yaxis_title="Δ Yield (percentage points)",
    template="plotly_white",
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Diagnostics (slope/curvature)
# -----------------------------
def interp(curve_terms, curve_vals, tgt):
    return np.interp(tgt, curve_terms, curve_vals)

sA2, sA10 = interp(terms_years, valsA, 2), interp(terms_years, valsA, 10)
sB2, sB10 = interp(terms_years, valsB, 2), interp(terms_years, valsB, 10)
slopeA = sA10 - sA2
slopeB = sB10 - sB2
slope_change = slopeB - slopeA

# simple curvature proxy using 5Y,10Y vs 2Y
cA = (interp(terms_years, valsA, 10) - interp(terms_years, valsA, 2)) - (interp(terms_years, valsA, 5) - interp(terms_years, valsA, 2))
cB = (interp(terms_years, valsB, 10) - interp(terms_years, valsB, 2)) - (interp(terms_years, valsB, 5) - interp(terms_years, valsB, 2))
curv_change = cB - cA

st.subheader("📌 Curve Diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("2Y (A → B)", f"{sA2:.2f}% → {sB2:.2f}%", f"{(sB2 - sA2):+.2f} pp")
m2.metric("10Y (A → B)", f"{sA10:.2f}% → {sB10:.2f}%", f"{(sB10 - sA10):+.2f} pp")
m3.metric("Slope (10Y–2Y)", f"{slopeA:.2f}% → {slopeB:.2f}%", f"{slope_change:+.2f} pp")
m4.metric("Curvature Δ", f"{cA:.2f} → {cB:.2f}", f"{curv_change:+.2f}")

# -----------------------------
# Auto‑Insights
# -----------------------------
def insights_text(slope_change, sA2, sB2, sA10, sB10):
    msgs = []
    if slope_change > 0.10:
        msgs.append("**Steepening:** Long yields rose vs short; markets may be pricing stronger growth/inflation or slower easing.")
    elif slope_change < -0.10:
        msgs.append("**Flattening/Inversion:** Short yields rose vs long (or fell less), pointing to tighter near‑term policy/growth concerns.")
    else:
        msgs.append("**Little slope change:** Overall shape broadly unchanged.")

    if (sB2 - sA2) > 0.10 and (sB10 - sA10) > 0:
        msgs.append("**Broad shift up:** Funding costs higher across maturities.")
    elif (sB2 - sA2) < -0.10 and (sB10 - sA10) < 0:
        msgs.append("**Broad shift down:** Borrowing conditions may ease if it persists.")
    else:
        msgs.append("**Mixed moves:** Short and long ends moved differently; watch credit spreads/issuance windows.")

    if slope_change > 0:
        msgs.append("- **Borrowers:** Consider terming out debt before long‑end yields climb further.")
        msgs.append("- **Investors:** Steepeners (long 10Y vs short 2Y) can benefit if the trend continues.")
    else:
        msgs.append("- **Borrowers:** Short‑dated funding may remain costly; hedge near‑term cash flows.")
        msgs.append("- **Investors:** Flatteners may benefit; be selective with rate‑sensitive sectors.")
    return "\n".join([f"- {m}" for m in msgs])

st.subheader("🧠 Auto‑Insights")
st.markdown(insights_text(slope_change, sA2, sB2, sA10, sB10))

st.caption(
    "Source: Bank of Canada Valet API — Selected/benchmark Government of Canada bond yields. "
    "For methodology and update timing, see BoC interest‑rate statistics and Valet API docs."
)

