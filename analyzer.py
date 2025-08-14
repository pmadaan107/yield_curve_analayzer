# app.py
# 📈 Canadian Yield Curve Analyzer (Streamlit)
# Features:
# - Fetches Canada zero-coupon daily yield curves (0.25Y ... 30Y)
# - Compare two dates/windows; detect steepening/flattening
# - Auto-insights for business & portfolio implications
# - Fallback: upload BoC zero-coupon CSV and analyze
# ---------------------------------------------------


import io
import datetime as dt
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Canadian Yield Curve Analyzer", layout="wide")
st.title("🇨🇦 Canadian Yield Curve Analyzer")
st.caption("Zero-coupon term structure • 0.25 y → 30 y • Source: Bank of Canada (live)")

# URL for BoC zero-coupon curves CSV
CSV_URL = "https://www.bankofcanada.ca/valet/observations/group/zero_coupon_curves/csv"

@st.cache_data(ttl=3600)
def fetch_zero_coupon():
    try:
        resp = requests.get(CSV_URL, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        df.columns = [c.strip() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        term_cols = [c for c in df.columns if _is_float(c)]
        df = df[term_cols]
        med = np.nanmedian(df.values.astype(float))
        if med < 1:
            df *= 100.0
        df.columns = [f"{float(c):.2f}y" for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def _is_float(s):
    try:
        float(s)
        return True
    except:
        return False

df = fetch_zero_coupon()
if df.empty:
    st.stop()

dates = df.index
min_date, max_date = dates.min().date(), dates.max().date()

st.sidebar.header("Controls")
mode = st.sidebar.radio("Compare", ["Two dates", "Window averages"], index=0)

# [— Insert rest of comparison logic, plotting, diagnostics, and insights here —]

# Reminder: We’ve removed upload/download functionality.
# Code should continue with plotting and insights as before.

# -----------------------------
# UI Header
# -----------------------------
st.title("🇨🇦 Canadian Yield Curve Analyzer")
st.caption("Zero‑coupon term structure • 0.25y → 30y • Source: Bank of Canada")

# -----------------------------
# Helper: Load BoC Zero-Coupon Yield Curves
# -----------------------------
# BoC provides daily zero-coupon curves downloadable as machine-readable files.
# Reference: BoC Valet API & yield curve data pages.
# If direct fetch fails (e.g., firewall), use the Upload section below.
#
# Implementation note:
# BoC's zero-coupon files are provided as wide CSVs with columns for each term
# from 0.25 to 30.00 years. We'll support both live fetch (if you map a URL)
# and user-upload.
#
# To keep this app robust, we provide a "plug-in" function you can point at a
# confirmed CSV endpoint. For now, we'll try the official page hint and then
# fall back to upload.

def parse_boc_zero_coupon_csv(content: bytes) -> pd.DataFrame:
    """Parse BoC zero-coupon CSV (wide format: Date + 0.25 ... 30.00 columns)."""
    df = pd.read_csv(io.BytesIO(content))
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # Identify date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        # Try first column as date
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Keep only numeric term columns (e.g., '0.25','0.50',...,'30.00')
    term_cols = []
    for c in df.columns:
        try:
            float(c)
            term_cols.append(c)
        except Exception:
            continue
    df = df[term_cols]
    # Express as percentages if given in decimals (0.05 → 5.0)
    # Heuristic: if median < 1, assume decimals; multiply by 100
    med = np.nanmedian(df.values.astype(float))
    if med is not np.nan and med < 1:
        df = df * 100.0
    # Rename columns to like "0.25y"
    df.columns = [f"{float(c):.2f}y" for c in df.columns]
    return df

def try_fetch_boc_zero_coupon() -> pd.DataFrame | None:
    """
    Attempt to fetch BoC zero-coupon curves.
    NOTE: The BoC site provides downloadable files from their 'Yield curves for zero-coupon bonds' page.
    If you have a direct CSV URL, place it in CSV_URLS below. Otherwise, rely on upload.
    """
    CSV_URLS = [
        # 👉 If you have a direct CSV path from BoC, put it here, e.g.:
        # "https://www.bankofcanada.ca/path/to/zero_coupon_yield_curves.csv"
    ]
    for url in CSV_URLS:
        try:
            r = requests.get(url, timeout=20)
            if r.ok and len(r.content) > 1000:
                return parse_boc_zero_coupon_csv(r.content)
        except Exception:
            continue
    return None

# -----------------------------
# Data Ingestion: Live fetch or Upload
# -----------------------------
with st.expander("📥 Data source", expanded=True):
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown(
            "This app uses **Bank of Canada** zero‑coupon yield curves (daily, 0.25y–30y). "
            "If live download is blocked, upload the CSV from "
            "[BoC’s zero‑coupon curves page](https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/)."
        )
    with colB:
        do_live = st.toggle("Try live download first", value=False)

    df_zero = None
    if do_live:
        with st.spinner("Trying live fetch from BoC…"):
            df_zero = try_fetch_boc_zero_coupon()
            if df_zero is not None:
                st.success("Live data loaded.")

    uploaded = st.file_uploader("Or upload the BoC zero‑coupon curve CSV (wide format)", type=["csv"])
    if uploaded is not None:
        try:
            df_zero = parse_boc_zero_coupon_csv(uploaded.read())
            st.success("File uploaded & parsed.")
        except Exception as e:
            st.error(f"Could not parse the file: {e}")

if df_zero is None or df_zero.empty:
    st.info(
        "No data yet. Either enable live fetch (and add a BoC CSV URL in the code) "
        "or download the CSV from the BoC page and upload it here."
    )
    st.stop()

# -----------------------------
# Controls
# -----------------------------
dates = df_zero.index
min_date, max_date = dates.min().date(), dates.max().date()
st.sidebar.header("🔧 Controls")

mode = st.sidebar.radio("Compare", ["Two specific dates", "Two windows (averages)"], index=0)
if mode == "Two specific dates":
    d1 = st.sidebar.date_input("Date A", value=max_date - dt.timedelta(days=30), min_value=min_date, max_value=max_date)
    d2 = st.sidebar.date_input("Date B", value=max_date, min_value=min_date, max_value=max_date)
    # Find nearest available rows
    idx1 = dates.get_indexer([pd.Timestamp(d1)], method="nearest")[0]
    idx2 = dates.get_indexer([pd.Timestamp(d2)], method="nearest")[0]
    curveA = df_zero.iloc[idx1]
    curveB = df_zero.iloc[idx2]
    labelA = df_zero.index[idx1].date().isoformat()
    labelB = df_zero.index[idx2].date().isoformat()
else:
    # Window averages
    window_days = st.sidebar.slider("Window length (days)", 5, 90, 30)
    endA = st.sidebar.date_input("Window A end", value=max_date - dt.timedelta(days=30), min_value=min_date, max_value=max_date)
    endB = st.sidebar.date_input("Window B end", value=max_date, min_value=min_date, max_value=max_date)
    startA = (pd.Timestamp(endA) - pd.Timedelta(days=window_days-1)).date()
    startB = (pd.Timestamp(endB) - pd.Timedelta(days=window_days-1)).date()

    winA = df_zero.loc[str(startA):str(endA)]
    winB = df_zero.loc[str(startB):str(endB)]

    curveA = winA.mean(numeric_only=True)
    curveB = winB.mean(numeric_only=True)
    labelA = f"{startA}→{endA}"
    labelB = f"{startB}→{endB}"

terms = [float(c.replace("y","")) for c in curveA.index]
order = np.argsort(terms)
terms_sorted = np.array(terms)[order]
A_vals = curveA.values[order]
B_vals = curveB.values[order]
diff = B_vals - A_vals

# -----------------------------
# Plots
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=terms_sorted, y=A_vals, mode="lines+markers",
    name=f"Curve A ({labelA})"
))
fig.add_trace(go.Scatter(
    x=terms_sorted, y=B_vals, mode="lines+markers",
    name=f"Curve B ({labelB})"
))
fig.update_layout(
    title="Zero‑Coupon Yield Curves (%, annualized)",
    xaxis_title="Maturity (years)",
    yaxis_title="Yield (%)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig, use_container_width=True)

# Diff chart
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=terms_sorted, y=diff, name="B - A (bps)"))
fig2.update_layout(
    title=f"Change in Yields (Curve B − Curve A)",
    xaxis_title="Maturity (years)",
    yaxis_title="Δ Yield (percentage points)",
    template="plotly_white",
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Diagnostics: Slope / Curvature
# -----------------------------
def slope_short_long(terms, curve, short_h=2.0, long_h=10.0):
    # linear interp at 2y and 10y (approx)
    return np.interp(short_h, terms, curve), np.interp(long_h, terms, curve)

sA2, sA10 = slope_short_long(terms_sorted, A_vals)
sB2, sB10 = slope_short_long(terms_sorted, B_vals)

slopeA = sA10 - sA2
slopeB = sB10 - sB2
slope_change = slopeB - slopeA  # >0 = steepening, <0 = flattening

# curvature proxy: (10y - 2y) - (5y - 2y) ~ hump
cA = (np.interp(10, terms_sorted, A_vals) - np.interp(2, terms_sorted, A_vals)) - (np.interp(5, terms_sorted, A_vals) - np.interp(2, terms_sorted, A_vals))
cB = (np.interp(10, terms_sorted, B_vals) - np.interp(2, terms_sorted, B_vals)) - (np.interp(5, terms_sorted, B_vals) - np.interp(2, terms_sorted, B_vals))
curv_change = cB - cA

st.subheader("📌 Curve Diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("2Y (A → B)", f"{sA2:.2f}% → {sB2:.2f}%", f"{(sB2 - sA2):+.2f} pp")
m2.metric("10Y (A → B)", f"{sA10:.2f}% → {sB10:.2f}%", f"{(sB10 - sA10):+.2f} pp")
m3.metric("Slope (10Y–2Y)", f"{slopeA:.2f}% → {slopeB:.2f}%", f"{slope_change:+.2f} pp")
m4.metric("Curvature Δ", f"{cA:.2f} → {cB:.2f}", f"{curv_change:+.2f}")

# -----------------------------
# Auto-Insights (Plain English)
# -----------------------------
def insights_text(slope_change, sA2, sB2, sA10, sB10):
    msgs = []
    if slope_change > 0.10:
        msgs.append("**Steepening:** Long rates rose vs short rates. Markets may be pricing stronger growth/inflation or a later pace of easing.")
    elif slope_change < -0.10:
        msgs.append("**Flattening/Inversion:** Short rates rose vs long (or fell less). Signals tighter near‑term policy or growth concerns.")
    else:
        msgs.append("**Little slope change:** The curve’s overall shape is broadly unchanged.")

    # Directional comments
    if (sB2 - sA2) > 0.10 and (sB10 - sA10) > 0:
        msgs.append("**Broad shift up:** Funding costs likely higher across maturities.")
    elif (sB2 - sA2) < -0.10 and (sB10 - sA10) < 0:
        msgs.append("**Broad shift down:** Borrowing conditions may ease if this persists.")
    else:
        msgs.append("**Mixed moves:** Short and long ends moved differently; watch credit spreads and issuance windows.")

    # Business & investor angles
    if slope_change > 0:
        msgs.append("- **Borrowers:** Consider terming out debt before long‑end yields climb further.")
        msgs.append("- **Investors:** Steepeners (long 10Y vs short 2Y) may benefit if the trend continues.")
    else:
        msgs.append("- **Borrowers:** Short‑dated funding may remain costly; hedge near‑term cash flows.")
        msgs.append("- **Investors:** Flatteners may benefit; consider rate‑sensitive sectors carefully.")

    return "\n".join([f"- {m}" for m in msgs])

st.subheader("🧠 Auto‑Insights")
st.markdown(insights_text(slope_change, sA2, sB2, sA10, sB10))

# -----------------------------
# Download: Clean Curves & Change
# -----------------------------
st.subheader("⬇️ Downloads")
out = pd.DataFrame({
    "term_years": terms_sorted,
    f"curve_A_{labelA}": A_vals,
    f"curve_B_{labelB}": B_vals,
    "delta_B_minus_A_pp": diff
})
st.dataframe(out, use_container_width=True)
st.download_button("Download CSV (curves + change)", out.to_csv(index=False), file_name="yield_curve_comparison.csv", mime="text/csv")

st.caption("Source: Bank of Canada zero‑coupon yield curves & Valet API.")
