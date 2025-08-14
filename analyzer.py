import datetime as dt
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Yield Curve (CSV direct)", layout="wide")
st.title("🇨🇦 Canadian Yield Curve Analyzer (CSV Direct)")
st.caption("Short end: Money market (T-bills). Long end: Benchmark bonds. Source: Bank of Canada CSV endpoints.")

# ---- CSV endpoints from BoC pages (no API key needed) ----
URL_TBILLS_CSV = "https://www.bankofcanada.ca/rates/interest-rates/t-bill-yields/csv"
URL_BENCH_CSV  = "https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/csv"

# Target labels exactly as they appear on pages
TBILL_TARGETS = {
    "0.08Y": "1 month",
    "0.25Y": "3 month",
    "0.50Y": "6 month",
    "1Y":    "1 year",
    # If 1-month is discontinued, it’ll just be missing—app will still work.
}
BOND_TARGETS = {
    "2Y":  "2 year",
    "3Y":  "3 year",
    "5Y":  "5 year",
    "7Y":  "7 year",
    "10Y": "10 year",
    "30Y": "Long-term",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    # BoC CSVs are standard; first column is typically Date
    df = pd.read_csv(io.BytesIO(r.content))
    # Normalize headers
    df.columns = [c.strip() for c in df.columns]
    # Find date column
    date_col = next((c for c in df.columns if c.lower().startswith("date")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    # Convert numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_columns_by_label(df: pd.DataFrame, wanted: dict) -> pd.DataFrame:
    # Map visible column labels (e.g., "3 month") to our curve labels (e.g., "0.25Y")
    out = {}
    for curve_label, page_label in wanted.items():
        if page_label in df.columns:
            out[curve_label] = df[page_label]
    return pd.DataFrame(out)

# ---- Load both CSVs ----
with st.spinner("Fetching Treasury bill CSV…"):
    tb = fetch_csv(URL_TBILLS_CSV)
with st.spinner("Fetching Benchmark bonds CSV…"):
    bench = fetch_csv(URL_BENCH_CSV)

tb_sel    = pick_columns_by_label(tb, TBILL_TARGETS)
bench_sel = pick_columns_by_label(bench, BOND_TARGETS)

if tb_sel.empty and bench_sel.empty:
    st.error("No data returned from the BoC CSV pages. Try again shortly.")
    st.stop()

df_raw = pd.concat([tb_sel, bench_sel], axis=1).sort_index().dropna(how="all")
if df_raw.empty:
    st.error("Data loaded, but no overlapping dates with values. Try changing date windows.")
    st.stop()

# ---- Controls ----
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

# ---- Prepare for plotting ----
def term_to_years(label: str) -> float:
    return float(label[:-1]) if label.endswith("Y") else np.nan

cols = [c for c in df_raw.columns if c in curveA.index and c in curveB.index]
terms = np.array([term_to_years(c) for c in cols], dtype=float)
mask = ~np.isnan(terms)
terms, yA, yB = terms[mask], curveA[mask].values, curveB[mask].values
order = np.argsort(terms)
x_terms, yA, yB = terms[order], yA[order], yB[order]

if len(x_terms) < 4:
    st.error("Not enough points to draw the curve. Try different dates/windows.")
    st.stop()

# ---- Charts ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_terms, y=yA, mode="lines+markers", name=f"Curve A ({labelA})"))
fig.add_trace(go.Scatter(x=x_terms, y=yB, mode="lines+markers", name=f"Curve B ({labelB})"))
fig.update_layout(title="Yield Curve (T-bills + Benchmarks), %", xaxis_title="Maturity (years)", yaxis_title="Yield (%)", template="plotly_white", legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

deltas = yB - yA
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=x_terms, y=deltas, name="B − A (pp)"))
fig2.update_layout(title="Change in Yields (Curve B − Curve A)", xaxis_title="Maturity (years)", yaxis_title="Δ Yield (percentage points)", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# ---- Diagnostics (2Y–10Y slope; 5Y hump curvature) ----
interp = lambda X, Y, t: float(np.interp(t, X, Y))
def safe_interp(t): 
    return (interp(x_terms, yA, t), interp(x_terms, yB, t)) if (x_terms.min() <= t <= x_terms.max()) else (np.nan, np.nan)

a2, b2 = safe_interp(2)
a10, b10 = safe_interp(10)
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
m1.metric("2Y (A → B)", f"{a2:.2f}% → {b2:.2f}%" if np.isfinite(a2) and np.isfinite(b2) else "—", f"{(b2-a2):+.2f} pp" if np.isfinite(a2) and np.isfinite(b2) else "")
m2.metric("10Y (A → B)", f"{a10:.2f}% → {b10:.2f}%" if np.isfinite(a10) and np.isfinite(b10) else "—", f"{(b10-a10):+.2f} pp" if np.isfinite(a10) and np.isfinite(b10) else "")
m3.metric("Slope (10Y–2Y)", f"{slopeA:.2f}% → {slopeB:.2f}%" if np.isfinite(slopeA) and np.isfinite(slopeB) else "—", f"{slope_change:+.2f} pp" if np.isfinite(slope_change) else "")
m4.metric("Curvature Δ", f"{cA:.2f} → {cB:.2f}" if np.isfinite(cA) and np.isfinite(cB) else "—", f"{curv_change:+.2f}" if np.isfinite(curv_change) else "")

# ---- Insights ----
def insights(slope_change, a2, b2, a10, b10):
    msgs = []
    if np.isfinite(slope_change):
        if slope_change > 0.10: msgs.append("**Steepening**: long end rose vs short (growth/inflation or slower easing).")
        elif slope_change < -0.10: msgs.append("**Flattening/Inversion**: short rose vs long (tighter policy/growth concerns).")
        else: msgs.append("**Little slope change**.")
    else:
        msgs.append("Slope insight unavailable (need both 2Y & 10Y).")

    if all(np.isfinite(x) for x in [a2, b2, a10, b10]):
        if (b2 - a2) > 0.10 and (b10 - a10) > 0: msgs.append("**Broad shift up**: funding costs higher across maturities.")
        elif (b2 - a2) < -0.10 and (b10 - a10) < 0: msgs.append("**Broad shift down**: borrowing conditions may ease.")
        else: msgs.append("**Mixed moves**: watch credit spreads/issuance windows.")
    else:
        msgs.append("Directional move insight limited.")
    return "\n".join(f"- {m}" for m in msgs)

st.subheader("🧠 Auto-Insights")
st.markdown(insights(slope_change, a2, b2, a10, b10))

st.caption("BoC sources: Treasury bill yields & Selected benchmark bond yields pages (CSV downloads).")
