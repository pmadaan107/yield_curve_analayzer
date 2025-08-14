# analyzer.py / app.py
# 🇨🇦 Canadian Yield Curve Analyzer — resilient fetch
# - Always fetches GoC benchmark (par) yields via BoC Valet (2Y,3Y,5Y,7Y,10Y,Long)
# - Tries to add T-bills via BoC CSV; if CSV fails (HTTPError/etc.), continues with bonds only
# - Compare two dates or two rolling windows; charts, diagnostics, insights

import datetime as dt
import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Yield Curve (Live, Resilient)", layout="wide")
st.title("🇨🇦 Canadian Yield Curve Analyzer (Live, Resilient)")
st.caption("Primary source: Bank of Canada Valet (benchmark bonds). Optional short end: T-bill CSV if available.")

# -----------------------------
# CONFIG
# -----------------------------
# Valet benchmark bond series (these are stable)
BOND_SERIES: Dict[str, str] = {
    "2Y":  "BD.CDN.2YR.DQ.YLD",
    "3Y":  "BD.CDN.3YR.DQ.YLD",
    "5Y":  "BD.CDN.5YR.DQ.YLD",
    "7Y":  "BD.CDN.7YR.DQ.YLD",
    "10Y": "BD.CDN.10YR.DQ.YLD",
    "30Y": "BD.CDN.LONG.DQ.YLD",
}
VALET_BASE = "https://www.bankofcanada.ca/valet/observations"

# Optional CSV (may be blocked in some environments). We’ll try it and fail gracefully.
URL_TBILLS_CSV = "https://www.bankofcanada.ca/rates/interest-rates/t-bill-yields/csv"
TBILL_TARGETS = {
    "0.08Y": "1 month",   # (If discontinued, it’ll just be missing.)
    "0.25Y": "3 month",
    "0.50Y": "6 month",
    "1Y":    "1 year",
}
VALET_GROUP_BASE = "https://www.bankofcanada.ca/valet/observations/group"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_valet_group(group_name: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Fetch a Valet group (JSON) → wide DF + {series_id: label} map."""
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

def pick_tbills_from_group(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Select 3m, 6m, 12m T-bills by label (robust to series IDs)."""
    want = {
        "0.25Y": ["treasury bills", "3 month"],
        "0.50Y": ["treasury bills", "6 month"],
        "1Y":    ["treasury bills", "1 year", "12 month"],
    }
    out = {}
    for curve_lbl, needles in want.items():
        sid = next((sid for sid, lab in labels.items()
                    if all(n in lab.lower() for n in needles)), None)
        if sid in df.columns:
            out[curve_lbl] = df[sid]
    return pd.DataFrame(out)


HEADERS = {
    # Some servers block default python UA; present as a browser
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/csv,application/json,application/*;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# -----------------------------
# Fetchers
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_valet_series(series_codes: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch multiple Valet series (JSON) → wide DF indexed by date. Columns are series codes."""
    frames = []
    for code in series_codes:
        url = f"{VALET_BASE}/{code}/json"
        params = {"start_date": start, "end_date": end}
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=25)
            if not r.ok:
                st.warning(f"Valet HTTP {r.status_code} for {code}: {r.text[:120]}")
                continue
            j = r.json()
            obs = j.get("observations", [])
            if not obs:
                st.warning(f"No observations for {code}.")
                continue
            dates = pd.to_datetime([o["d"] for o in obs])
            vals = [float(o[code]["v"]) if o.get(code) and o[code].get("v") not in (None, "") else np.nan for o in obs]
            frames.append(pd.Series(vals, index=dates, name=code))
        except Exception as e:
            st.warning(f"Error fetching {code}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tbills_csv(url: str) -> pd.DataFrame:
    """Try to fetch the BoC T-bill CSV; return empty DF on any error (don’t crash)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        date_col = next((c for c in df.columns if c.lower().startswith("date")), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        # force numeric
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        st.info(f"T-bill CSV not available right now ({type(e).__name__}). Continuing with bonds only.")
        return pd.DataFrame()

def select_tbills(df_tbill: pd.DataFrame, wanted: Dict[str, str]) -> pd.DataFrame:
    if df_tbill.empty:
        return pd.DataFrame()
    out = {}
    for curve_label, page_label in wanted.items():
        if page_label in df_tbill.columns:
            out[curve_label] = df_tbill[page_label]
    return pd.DataFrame(out)

# -----------------------------
# Load data window
# -----------------------------
today = dt.date.today()
start_default = (today - dt.timedelta(days=3*365)).isoformat()
end_default = today.isoformat()

with st.spinner("Fetching benchmark bond yields (Valet)…"):
    bonds_df = fetch_valet_series(list(BOND_SERIES.values()), start_default, end_default)

if bonds_df.empty:
    st.error("No benchmark bond data from Valet. Please retry later.")
    st.stop()

# map Valet codes → maturity labels
bonds_df = bonds_df.rename(columns={v: k for k, v in BOND_SERIES.items()})

with st.spinner("Fetching T-bill short end (CSV)…"):
    tbills_raw = fetch_tbills_csv(URL_TBILLS_CSV)
tbills_df = select_tbills(tbills_raw, TBILL_TARGETS)  # may be empty

# Merge curve parts
df_raw = pd.concat([tbills_df, bonds_df], axis=1).sort_index().dropna(how="all")
if df_raw.empty:
    # Fallback to bonds only (if tbills caused all-NaN unions)
    df_raw = bonds_df.copy()

# -----------------------------
# Controls
# -----------------------------
st.sidebar.header("🔧 Controls")
mode = st.sidebar.radio("Compare", ["Two specific dates", "Two windows (averages)"], index=0)

min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()

def nearest_row(df: pd.DataFrame, d: dt.date) -> Tuple[pd.Series, str]:
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
# Prepare plotting arrays
# -----------------------------
def term_to_years(lbl: str) -> float:
    try:
        if lbl.endswith("Y"):
            return float(lbl[:-1])
        return np.nan
    except Exception:
        return np.nan

cols = [c for c in df_raw.columns if c in curveA.index and c in curveB.index]
terms = np.array([term_to_years(c) for c in cols], dtype=float)
mask = ~np.isnan(terms)
terms = terms[mask]
yA = np.array([curveA[c] for c in np.array(cols)[mask]], dtype=float)
yB = np.array([curveB[c] for c in np.array(cols)[mask]], dtype=float)

order = np.argsort(terms)
x_terms, yA, yB = terms[order], yA[order], yB[order]

if len(x_terms) < 3:
    st.error("Not enough points to draw the curve (need ≥3 maturities). Try different dates/windows.")
    st.stop()

# -----------------------------
# Charts
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_terms, y=yA, mode="lines+markers", name=f"Curve A ({labelA})"))
fig.add_trace(go.Scatter(x=x_terms, y=yB, mode="lines+markers", name=f"Curve B ({labelB})"))
fig.update_layout(
    title="Yield Curve (Benchmark bonds + T-bills when available)",
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
# Diagnostics & Insights
# -----------------------------
def interp(x, y, t):
    return float(np.interp(t, x, y))

def safe_interp(pair_t) -> Tuple[float, float]:
    try:
        return interp(x_terms, yA, pair_t), interp(x_terms, yB, pair_t)
    except Exception:
        return np.nan, np.nan

a2, b2 = safe_interp(2.0)
a10, b10 = safe_interp(10.0)
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
            msgs.append("**Little slope change.**")
    else:
        msgs.append("Slope insight unavailable (need both 2Y & 10Y).")

    if all(np.isfinite(x) for x in [a2, b2, a10, b10]):
        if (b2 - a2) > 0.10 and (b10 - a10) > 0:
            msgs.append("**Broad shift up:** Funding costs higher across maturities.")
        elif (b2 - a2) < -0.10 and (b10 - a10) < 0:
            msgs.append("**Broad shift down:** Borrowing conditions may ease if this persists.")
        else:
            msgs.append("**Mixed moves:** Watch credit spreads and issuance windows.")
    else:
        msgs.append("Directional move insight limited (missing points).")
    return "\n".join(f"- {m}" for m in msgs)

st.subheader("🧠 Auto-Insights")
st.markdown(insights(slope_change, a2, b2, a10, b10))

st.caption("Primary: Bank of Canada Valet benchmark bond yields (live). "
           "Optional: BoC T-bill CSV (short end) if accessible; app continues without it when blocked.")

