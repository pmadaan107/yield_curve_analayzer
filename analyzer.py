# app.py
# 🇨🇦 Canadian Yield Curve Analyzer — Live via Bank of Canada Valet API (Groups)
# - Short end: T-bills (1M, 2M, 3M, 6M, 1Y) from group "money_market"
# - Long end: Benchmark bonds (2Y, 3Y, 5Y, 7Y, 10Y, Long≈30Y) from group "bond_yields_benchmark"
# - Compare two dates or two rolling windows; charts & insights

import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Canadian Yield Curve (Live)", layout="wide")
st.title("🇨🇦 Canadian Yield Curve Analyzer (Live)")
st.caption("Short end: Money market (T-bills). Long end: Benchmark bonds. Source: Bank of Canada Valet API.")

VALET_BASE = "https://www.bankofcanada.ca/valet/observations/group"

# Which labels to look for inside each group (robust to internal series IDs)
TBILL_TARGETS = {
    "0.08Y": "Treasury bills - 1 month",   # 1/12 ≈ 0.08y
    "0.17Y": "Treasury bills - 2 month",   # 2/12 ≈ 0.17y
    "0.25Y": "Treasury bills - 3 month",
    "0.50Y": "Treasury bills - 6 month",
    "1Y":    "Treasury bills - 1 year",
}

BENCH_TARGETS = {
    "2Y":   "Government of Canada benchmark bond yields - 2 year",
    "3Y":   "Government of Canada benchmark bond yields - 3 year",
    "5Y":   "Government of Canada benchmark bond yields - 5 year",
    "7Y":   "Government of Canada benchmark bond yields - 7 year",
    "10Y":  "Government of Canada benchmark bond yields - 10 year",
    "30Y":  "Government of Canada benchmark bond yields - Long-term",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_group(group_name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch a Valet *group* as a wide DataFrame; columns will be the series IDs."""
    params = {}
    if start: params["start_date"] = start
    if end:   params["end_date"] = end
    url = f"{VALET_BASE}/{group_name}/json"
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    js = r.json()

    # observations: list of dicts with 'd' and per-series keys
    obs = js.get("observations", [])
    if not obs:
        return pd.DataFrame()

    dates = pd.to_datetime([o["d"] for o in obs])
    # Build columns by series id present in seriesDetail
    detail = js.get("seriesDetail", {})
    series_ids = list(detail.keys())
    data = {}
    for sid in series_ids:
        vals = []
        for o in obs:
            cell = o.get(sid)
            vals.append(float(cell["v"]) if cell and cell.get("v") not in (None, "") else np.nan)
        data[sid] = pd.Series(vals, index=dates)

    df = pd.DataFrame(data).sort_index()
    # Attach a mapping sid -> label for later filtering
    labels = {sid: detail[sid].get("label", sid) for sid in series_ids}
    return df, labels

def select_by_label(df_labels_pair, wanted: dict) -> pd.DataFrame:
    """From (df, labels_map), select columns whose label matches each wanted label."""
    df, labels = df_labels_pair
    out = {}
    for target_maturity, wanted_label in wanted.items():
        # find first series id whose label matches exactly (case-sensitive as per BoC labels)
        sid = next((k for k, v in labels.items() if v == wanted_label), None)
        if sid and sid in df.columns:
            out[target_maturity] = df[sid]
    if not out:
        return pd.DataFrame()
    sel = pd.DataFrame(out)
    # Ensure numeric percent (Valet yields are already in %)
    return sel

# -----------------------------
# Pull last ~3 years for speed
# -----------------------------
today = dt.date.today()
start_default = (today - dt.timedelta(days=3*365)).isoformat()
end_default = today.isoformat()

with st.spinner("Fetching money market (T-bills)…"):
    tb_df, tb_labels = fetch_group("money_market", start_default, end_default)
tb = select_by_label((tb_df, tb_labels), TBILL_TARGETS)

with st.spinner("Fetching benchmark bonds…"):
    bench_df, bench_labels = fetch_group("bond_yields_benchmark", start_default, end_default)
bench = select_by_label((bench_df, bench_labels), BENCH_TARGETS)

# Merge: short + long
if tb.empty and bench.empty:
    st.error("No data returned from Valet groups. Please try again later.")
    st.stop()

df_raw = pd.concat([tb, bench], axis=1)
df_raw = df_raw.sort_index().dropna(how="all")
if df_raw.empty:
    st.error("No overlapping observations found for the selected series.")
    st.stop()

# -----------------------------
# Controls
# -----------------------------
st.sidebar.header("🔧 Controls")
mode = st.sidebar.radio("Compare", ["Two specific dates", "Two windows (averages)"], index=0)

min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()

def nearest_row(df: pd.DataFrame, d: dt.date) -> tuple[pd.Series, str]:
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

# -----------------------------
# Prepare terms for plotting
# -----------------------------
# Convert labels like "0.08Y", "0.17Y", ..., "30Y" to float years
def term_to_years(col: str) -> float:
    if col.endswith("Y"):
        return float(col[:-1])
    return np.nan

cols_present = [c for c in df_raw.columns if c in curveA.index and c in curveB.index]
terms = np.array([term_to_years(c) for c in cols_present], dtype=float)
mask = ~np.isnan(terms)
terms = terms[mask]
valsA = np.array([curveA[c] for c in np.array(cols_present)[mask]], dtype=float)
valsB = np.array([curveB[c] for c in np.array(cols_present)[mask]], dtype=float)

order = np.argsort(terms)
x_terms = terms[order]
yA = valsA[order]
yB = valsB[order]

if len(x_terms) < 4:
    st.error("Not enough points to draw the curve. Try different dates/windows.")
    st.stop()

# -----------------------------
# Charts
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_terms, y=yA, mode="lines+markers", name=f"Curve A ({labelA})"))
fig.add_trace(go.Scatter(x=x_terms, y=yB, mode="lines+markers", name=f"Curve B ({labelB})"))
fig.update_layout(
    title="Yield Curve (T-bills + Benchmarks), %",
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
# Diagnostics (focus on 2Y–10Y slope; robust if both available)
# -----------------------------
def interp(x, y, target):
    return float(np.interp(target, x, y))

def safe_interp(term, arr_x, arr_y):
    try:
        return interp(arr_x, arr_y, term)
    except Exception:
        return np.nan

sA2, sA10 = safe_interp(2, x_terms, yA), safe_interp(10, x_terms, yA)
sB2, sB10 = safe_interp(2, x_terms, yB), safe_interp(10, x_terms, yB)

slopeA = sA10 - sA2 if np.isfinite(sA2) and np.isfinite(sA10) else np.nan
slopeB = sB10 - sB2 if np.isfinite(sB2) and np.isfinite(sB10) else np.nan
slope_change = slopeB - slopeA if np.isfinite(slopeA) and np.isfinite(slopeB) else np.nan

# curvature proxy using 2Y, 5Y, 10Y if available
def curvature(x, y):
    if np.all(np.isfinite([safe_interp(2,x,y), safe_interp(5,x,y), safe_interp(10,x,y)])):
        return (interp(x,y,10) - interp(x,y,2)) - (interp(x,y,5) - interp(x,y,2))
    return np.nan

cA = curvature(x_terms, yA)
cB = curvature(x_terms, yB)
curv_change = cB - cA if np.isfinite(cA) and np.isfinite(cB) else np.nan

st.subheader("📌 Curve Diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("2Y (A → B)", f"{sA2:.2f}% → {sB2:.2f}%" if np.isfinite(sA2) and np.isfinite(sB2) else "—", 
          f"{(sB2 - sA2):+.2f} pp" if np.isfinite(sA2) and np.isfinite(sB2) else "")
m2.metric("10Y (A → B)", f"{sA10:.2f}% → {sB10:.2f}%" if np.isfinite(sA10) and np.isfinite(sB10) else "—", 
          f"{(sB10 - sA10):+.2f} pp" if np.isfinite(sA10) and np.isfinite(sB10) else "")
m3.metric("Slope (10Y–2Y)", f"{slopeA:.2f}% → {slopeB:.2f}%" if np.isfinite(slopeA) and np.isfinite(slopeB) else "—", 
          f"{slope_change:+.2f} pp" if np.isfinite(slope_change) else "")
m4.metric("Curvature Δ", f"{cA:.2f} → {cB:.2f}" if np.isfinite(cA) and np.isfinite(cB) else "—", 
          f"{curv_change:+.2f}" if np.isfinite(curv_change) else "")

# -----------------------------
# Auto-Insights
# -----------------------------
def insights_text(slope_change, sA2, sB2, sA10, sB10):
    msgs = []
    if np.isfinite(slope_change):
        if slope_change > 0.10:
            msgs.append("**Steepening:** Long yields rose vs short; markets may be pricing stronger growth/inflation or slower policy easing.")
        elif slope_change < -0.10:
            msgs.append("**Flattening/Inversion:** Short yields rose vs long (or fell less), pointing to tighter near-term policy or growth concerns.")
        else:
            msgs.append("**Little slope change:** Overall shape broadly unchanged.")
    else:
        msgs.append("Slope insight unavailable (need both 2Y and 10Y).")

    if np.isfinite(sB2) and np.isfinite(sA2) and np.isfinite(sB10) and np.isfinite(sA10):
        if (sB2 - sA2) > 0.10 and (sB10 - sA10) > 0:
            msgs.append("**Broad shift up:** Funding costs higher across maturities.")
        elif (sB2 - sA2) < -0.10 and (sB10 - sA10) < 0:
            msgs.append("**Broad shift down:** Borrowing conditions may ease if this persists.")
        else:
            msgs.append("**Mixed moves:** Short and long ends moved differently; watch credit spreads and issuance windows.")
    else:
        msgs.append("Directional move insight limited (missing 2Y/10Y).")

    if np.isfinite(slope_change):
        if slope_change > 0:
            msgs.append("- **Borrowers:** Consider terming out debt before long-end yields climb further.")
            msgs.append("- **Investors:** Steepener views (long 10Y vs short 2Y) may benefit if the trend continues.")
        else:
            msgs.append("- **Borrowers:** Short-dated funding may remain costly; hedge near-term cash flows.")
            msgs.append("- **Investors:** Flattener views can benefit; be selective with rate-sensitive sectors.")
    return "\n".join([f"- {m}" for m in msgs])

st.subheader("🧠 Auto-Insights")
st.markdown(insights_text(slope_change, sA2, sB2, sA10, sB10))

st.caption("Sources: Bank of Canada Valet groups — Money market yields (T-bills) and Benchmark bond yields. Data fetched live and cached for 1 hour.")


