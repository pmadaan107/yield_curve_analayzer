# app.py
# 🇨🇦 Canadian Yield Curve Analyzer — Bank of Canada (Valet groups)
# - Short end: money_market group (Treasury bills 3m, 6m, 12m)
# - Long end: bond_yields_benchmark group (2Y, 3Y, 5Y, 7Y, 10Y, 30Y)
# - Compare two dates or two rolling windows; charts + diagnostics + insights

import datetime as dt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Yield Curve (Bank of Canada)", layout="wide")
st.title("🇨🇦 Canadian Yield Curve — Bank of Canada (Valet)")
st.caption("Short end: Money market (T-bills). Long end: Benchmark bonds. Data fetched live from BoC Valet.")

VALET_GROUP_BASE = "https://www.bankofcanada.ca/valet/observations/group"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
}

# Which labels to select from each group (match is case-insensitive & ANDed)
TBILL_WANTS = {
    "0.25Y": ["treasury bills", "3 month"],
    "0.50Y": ["treasury bills", "6 month"],
    "1Y":    ["treasury bills", "1 year"],   # also matches “12 month” text
    # If 1-month exists and you want it, add:
    # "0.08Y": ["treasury bills", "1 month"]
}
BENCH_WANTS = {
    "2Y":   ["benchmark bond yields", "2 year"],
    "3Y":   ["benchmark bond yields", "3 year"],
    "5Y":   ["benchmark bond yields", "5 year"],
    "7Y":   ["benchmark bond yields", "7 year"],
    "10Y":  ["benchmark bond yields", "10 year"],
    "30Y":  ["benchmark bond yields", "long-term"],
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_valet_group(group_name: str, start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Fetch a BoC Valet *group* → (wide DF, {series_id: label})."""
    url = f"{VALET_GROUP_BASE}/{group_name}/json"
    params = {"start_date": start, "end_date": end}
    r = requests.get(url, params=params, headers=HEADERS, timeout=25)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    detail = js.get("seriesDetail", {})
    if not obs or not detail:
        return pd.DataFrame(), {}

    dates = pd.to_datetime([o["d"] for o in obs])
    data = {}
    for sid in detail.keys():
        vals = [float(o[sid]["v"]) if o.get(sid) and o[sid].get("v") not in (None, "") else np.nan for o in obs]
        data[sid] = pd.Series(vals, index=dates, name=sid)
    df = pd.DataFrame(data).sort_index()
    labels = {sid: detail[sid].get("label", sid) for sid in detail}
    return df, labels

def select_by_label(df: pd.DataFrame, labels: Dict[str, str], wants: Dict[str, list]) -> pd.DataFrame:
    """Pick columns whose label contains ALL tokens in wants (case-insensitive)."""
    if df.empty or not labels:
        return pd.DataFrame()
    labels_lc = {sid: lab.lower() for sid, lab in labels.items()}
    out = {}
    for curve_label, tokens in wants.items():
        sid = next((sid for sid, lab in labels_lc.items() if all(tok in lab for tok in tokens)), None)
        if sid and sid in df.columns:
            out[curve_label] = df[sid]
    return pd.DataFrame(out)

# -----------------------------
# Load window (last ~3 years)
# -----------------------------
today = dt.date.today()
start_default = (today - dt.timedelta(days=3*365)).isoformat()
end_default = today.isoformat()

with st.spinner("Fetching money market (T-bills)…"):
    mm_df, mm_labels = fetch_valet_group("money_market", start_default, end_default)
tbills = select_by_label(mm_df, mm_labels, TBILL_WANTS)  # may be empty if data not present today

with st.spinner("Fetching benchmark bonds…"):
    bench_df, bench_labels = fetch_valet_group("bond_yields_benchmark", start_default, end_default)
bench = select_by_label(bench_df, bench_labels, BENCH_WANTS)

if tbills.empty and bench.empty:
    st.error("No data returned from Bank of Canada Valet (groups). Please try again later.")
    st.stop()

# Merge into a single curve (short + long)
df_raw = pd.concat([tbills, bench], axis=1).sort_index().dropna(how="all")
if df_raw.empty:
    # Fall back to whichever side has data
    df_raw = bench if not bench.empty else tbills

# -----------------------------
# Controls
# -----------------------------
st.sidebar.header("🔧 Controls")
mode = st.sidebar.radio("Compare", ["Two specific dates", "Two windows (averages)"], index=0)

min_date, max_date = df_raw.index.min().date(), df_raw.index.max().date()

def nearest_row(df: pd.DataFrame, d: dt.date):
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
    curveA = df_raw.loc[str(startA):str(endA)].mean(numeric_only=True)
    curveB = df_raw.loc[str(startB):str(endB)].mean(numeric_only=True)
    labelA, labelB = f"{startA}→{endA}", f"{startB}→{endB}"

# -----------------------------
# Prep for plotting
# -----------------------------
def label_to_years(lbl: str) -> float:
    try:
        if lbl.endswith("Y"):
            return float(lbl[:-1])
    except Exception:
        pass
    return np.nan

cols = [c for c in df_raw.columns if c in curveA.index and c in curveB.index]
terms = np.array([label_to_years(c) for c in cols], dtype=float)
mask = ~np.isnan(terms)
terms = terms[mask]
yA = np.array([curveA[c] for c in np.array(cols)[mask]], dtype=float)
yB = np.array([curveB[c] for c in np.array(cols)[mask]], dtype=float)

order = np.argsort(terms)
x_terms, yA, yB = terms[order], yA[order], yB[order]

if len(x_terms) < 3:
    st.error("Not enough maturities to draw the curve (need ≥ 3). Try different dates/windows.")
    st.stop()

# -----------------------------
# Charts
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_terms, y=yA, mode="lines+markers", name=f"Curve A ({labelA})"))
fig.add_trace(go.Scatter(x=x_terms, y=yB, mode="lines+markers", name=f"Curve B ({labelB})"))
fig.update_layout(
    title="Yield Curve (Money market + Benchmark bonds), %",
    xaxis_title="Maturity (years)",
    yaxis_title="Yield (%)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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
# Diagnostics & Insights
# -----------------------------
def interp(x, y, t):
    return float(np.interp(t, x, y))

def safe_pair(t):
    try:
        return interp(x_terms, yA, t), interp(x_terms, yB, t)
    except Exception:
        return np.nan, np.nan

a2, b2 = safe_pair(2.0)
a10, b10 = safe_pair(10.0)
slopeA = (a10 - a2) if np.isfinite(a2) and np.isfinite(a10) else np.nan
slopeB = (b10 - b2) if np.isfinite(b2) and np.isfinite(b10) else np.nan
slope_change = (slopeB - slopeA) if np.isfinite(slopeA) and np.isfinite(slopeB) else np.nan

def curvature(x, y):
    try:
        return (interp(x, y, 10) - interp(x, y, 2)) - (interp(x, y, 5) - interp(x, y, 2))
    except Exception:
        return np.nan

cA, cB = curvature(x_terms, yA), curvature(x_terms, yB)
curv_change = (cB - cA) if np.isfinite(cA) and np.isfinite(cB) else np.nan

st.subheader("📌 Curve Diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("2Y (A → B)", f"{a2:.2f}% → {b2:.2f}%" if np.isfinite(a2) and np.isfinite(b2) else "—",
          f"{(b2-a2):+.2f} pp" if np.isfinite(a2) and np.isfinite(b2) else "")
m2.metric("10Y (A → B)", f"{a10:.2f}% → {b10:.2f}%" if np.isfinite(a10) and np.isfinite(b10) else "—",
          f"{(b10-a10):+.2f} pp" if np.isfinite(a10) and np.isfinite(b10) else "")
m3.metric("Slope (10Y–2Y)", f"{slopeA:.2f}% → {slopeB:.2f}%" if np.isfinite(slopeA) and np.isfinite(slopeB) else "—",
          f"{slope_change:+.2f} pp" if np.isfinite(slope_change) else "")
m4.metric("Curvature Δ", f"{cA:.2f} → {cB:.2f}" if np.isfinite(cA) and np.isfinite(cB) else "—",
          f"{curv_change:+.2f}" if np.isfinite(curv_change) else "")

def insights(slope_change, a2, b2, a10, b10):
    msgs = []
    if np.isfinite(slope_change):
        if slope_change > 0.10:
            msgs.append("**Steepening:** Long end rose vs short; markets may be pricing stronger growth/inflation or slower easing.")
        elif slope_change < -0.10:
            msgs.append("**Flattening/Inversion:** Short end rose vs long (or fell less), pointing to tighter near-term policy or growth concerns.")
        else:
            msgs.append("**Little slope change** overall.")
    else:
        msgs.append("Slope insight unavailable (need both 2Y & 10Y).")

    if all(np.isfinite(x) for x in [a2, b2, a10, b10]):
        if (b2 - a2) > 0.10 and (b10 - a10) > 0:
            msgs.append("**Broad shift up:** Funding costs higher across maturities.")
        elif (b2 - a2) < -0.10 and (b10 - a10) < 0:
            msgs.append("**Broad shift down:** Borrowing conditions may ease.")
        else:
            msgs.append("**Mixed moves:** Watch credit spreads and issuance windows.")
    else:
        msgs.append("Directional insight limited (missing 2Y/10Y).")
    return "\n".join(f"- {m}" for m in msgs)

st.subheader("🧠 Auto-Insights")
st.markdown(insights(slope_change, a2, b2, a10, b10))

st.caption("Sources: Bank of Canada — Money market yields & Selected benchmark bond yields (JSON via Valet groups).")



