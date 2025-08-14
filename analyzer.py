# app.py
# 🌍 Global Risk Dashboard (Deloitte-ready skeleton)
# - Validates FRED key up-front (Step 1)
# - Graceful fallbacks to Yahoo Finance
# - KPIs + yield spread + FX + (optional) CPI/VIX
# - Clean, professional UI

import os
import time
from datetime import date, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# Optional deps (only used if FRED key is present)
try:
    from fredapi import Fred
except Exception:
    Fred = None

# =========================
# 1) FRED key: set & validate (fast fail + clear errors)
# =========================
FRED_KEY = os.getenv("FRED_API_KEY")

def fred_client():
    """Return a validated Fred client, or None if missing/invalid/unavailable."""
    if (Fred is None) or not FRED_KEY:
        return None
    try:
        fred = Fred(api_key=FRED_KEY)
        # Lightweight probe to validate key without pulling huge data
        _ = fred.get_series('DGS10', observation_start='2024-01-01').head(1)
        return fred
    except Exception as e:
        # Log to terminal; we also surface a small note in UI later
        print(f"[FRED init error] {e}")
        return None

fred = fred_client()

# =========================
# Utility: cache yfinance calls
# =========================
@st.cache_data(ttl=60 * 30)  # cache for 30 minutes
def yf_download(tickers, start=None, end=None, interval='1d'):
    return yf.download(tickers, start=start, end=end, interval=interval, progress=False)

# =========================
# 2) Robust data fetchers (with fallback)
# =========================
def fetch_us_yields(start='2020-01-01'):
    """
    Returns DataFrame with columns ['10Y','2Y'] (percent yields), daily index.
    Primary: FRED (DGS10, DGS2). Fallback: Yahoo (^TNX, ^UST2Y).
    """
    # --- Primary: FRED ---
    if fred is not None:
        for attempt in range(3):
            try:
                ten = fred.get_series('DGS10', observation_start=start)  # percent
                two = fred.get_series('DGS2',  observation_start=start)  # percent
                out = pd.DataFrame({'10Y': ten, '2Y': two}).dropna().astype(float)
                out.index = pd.to_datetime(out.index)
                return out
            except Exception as e:
                print(f"[FRED fetch attempt {attempt+1} error] {e}")
                time.sleep(1 + attempt)

    # --- Fallback: Yahoo Finance ---
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(date.today())
    y = yf_download(['^TNX', '^UST2Y'], start=start_dt, end=end_dt)['Close']

    # Normalize shape
    if isinstance(y, pd.Series):
        y = y.to_frame()

    # Heuristic normalization: these tickers are often yield*10 or yield*100 historically
    def normalize(series: pd.Series) -> pd.Series:
        s = series.dropna().copy()
        if s.empty:
            return s
        med = s.median()
        # If absurdly high, scale down. Typical modern US 10Y ~ 4-ish.
        if med > 20:
            s = s / 10.0
        return s

    tnx = normalize(y.get('^TNX', pd.Series(dtype=float)))
    ust2 = normalize(y.get('^UST2Y', pd.Series(dtype=float)))

    out = pd.DataFrame({'10Y': tnx, '2Y': ust2}).dropna()
    out.index = pd.to_datetime(out.index)
    return out

def fetch_fx_pair(ticker='CADUSD=X', period='1y', interval='1d'):
    """Returns a Series of FX close levels; falls back to empty Series on failure."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)['Close']
        data = data.dropna()
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        print(f"[FX fetch error] {e}")
        return pd.Series(dtype=float)

def fetch_vix(period='1y', interval='1d'):
    """Returns VIX close Series from Yahoo."""
    try:
        data = yf.download('^VIX', period=period, interval=interval, progress=False)['Close']
        data = data.dropna()
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        print(f"[VIX fetch error] {e}")
        return pd.Series(dtype=float)

def fetch_cpi_us(start='2015-01-01'):
    """
    Returns monthly US CPI index (CPIAUCSL) if FRED is available; else empty Series.
    Provides YoY inflation if possible.
    """
    if fred is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    try:
        cpi = fred.get_series('CPIAUCSL', observation_start=start)  # index level
        cpi = cpi.asfreq('MS')  # monthly
        cpi.index = pd.to_datetime(cpi.index)
        yoy = (cpi / cpi.shift(12) - 1) * 100.0
        return cpi, yoy
    except Exception as e:
        print(f"[CPI fetch error] {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

# =========================
# 3) Finance helpers
# =========================
def latest_value(series: pd.Series):
    if series is None or series.empty:
        return np.nan
    return float(series.dropna().iloc[-1])

def pct_change(series: pd.Series, days: int = 1):
    """Percentage change over N observations (not calendar days)."""
    if series is None or len(series.dropna()) <= days:
        return np.nan
    s = series.dropna()
    return float((s.iloc[-1] / s.iloc[-1 - days] - 1) * 100.0)

# =========================
# 4) Streamlit UI
# =========================
st.set_page_config(page_title="🌍 Global Risk Dashboard", layout="wide")
st.markdown("<h1 style='margin-bottom:0.25rem'>🌍 Global Risk Dashboard</h1>", unsafe_allow_html=True)
st.caption("Consulting-style snapshot of macro & market risk. FRED-backed with Yahoo fallbacks.")

# Small banner if FRED missing/invalid
if fred is None:
    st.warning("FRED API not available (missing or invalid key). Falling back to Yahoo where possible.")
else:
    st.success("FRED API connected successfully.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    lookback_years = st.slider("Lookback (years)", min_value=1, max_value=10, value=3, step=1)
    start_date = (date.today() - timedelta(days=365 * lookback_years)).isoformat()
    scenario = st.selectbox(
        "Scenario (applied to headline metrics)",
        ["Base (no shock)", "Recession Shock (-100bps rates, -20% equities, -5% FX)", "Inflation Spike (+150bps rates, +10% oil, +5% FX)"]
    )

# =========================
# 5) Load data
# =========================
yc = fetch_us_yields(start=start_date)
fx = fetch_fx_pair('CADUSD=X', period=f'{lookback_years}y', interval='1d')
vix = fetch_vix(period=f'{lookback_years}y', interval='1d')
cpi, cpi_yoy = fetch_cpi_us(start=start_date)

# Derive spread
yc_spread = pd.Series(dtype=float)
if not yc.empty and all(col in yc.columns for col in ['10Y', '2Y']):
    yc_spread = yc['10Y'] - yc['2Y']

# =========================
# 6) Scenario transforms (simple illustrative)
# =========================
def apply_scenario_to_kpi(kpi_name: str, value: float) -> float:
    if np.isnan(value):
        return value
    if scenario == "Base (no shock)":
        return value
    if scenario == "Recession Shock (-100bps rates, -20% equities, -5% FX)":
        if kpi_name in ("10Y", "2Y"):  # rates down 100 bps
            return value - 1.00
        if kpi_name == "CADUSD":       # FX down 5%
            return value * 0.95
        if kpi_name == "VIX":          # vol often spikes; simple +50% bump
            return value * 1.5
    if scenario == "Inflation Spike (+150bps rates, +10% oil, +5% FX)":
        if kpi_name in ("10Y", "2Y"):  # rates up 150 bps
            return value + 1.50
        if kpi_name == "CADUSD":       # FX up 5% (illustrative)
            return value * 1.05
        if kpi_name == "VIX":
            return value * 1.25
    return value

# =========================
# 7) KPIs
# =========================
col1, col2, col3, col4 = st.columns(4)

latest_10y = latest_value(yc['10Y']) if '10Y' in yc.columns else np.nan
latest_2y = latest_value(yc['2Y']) if '2Y' in yc.columns else np.nan
latest_spread = latest_value(yc_spread)
latest_fx = latest_value(fx)
latest_vix = latest_value(vix)
latest_infl = latest_value(cpi_yoy)

# Apply scenario to headline rate & FX & VIX (not to CPI YoY)
sc_10y = apply_scenario_to_kpi("10Y", latest_10y)
sc_2y  = apply_scenario_to_kpi("2Y", latest_2y)
sc_fx  = apply_scenario_to_kpi("CADUSD", latest_fx)
sc_vix = apply_scenario_to_kpi("VIX", latest_vix)
# Recompute spread post-scenario
sc_spread = sc_10y - sc_2y if not any(map(np.isnan, [sc_10y, sc_2y])) else np.nan

with col1:
    st.metric("US 10Y Yield (%)", f"{sc_10y:,.2f}" if not np.isnan(sc_10y) else "—",
              delta=f"{sc_10y - latest_10y:+.2f} vs base" if not any(map(np.isnan, [sc_10y, latest_10y])) else None)

with col2:
    st.metric("US 2Y Yield (%)", f"{sc_2y:,.2f}" if not np.isnan(sc_2y) else "—",
              delta=f"{sc_2y - latest_2y:+.2f} vs base" if not any(map(np.isnan, [sc_2y, latest_2y])) else None)

with col3:
    st.metric("10Y–2Y Spread (bps)", f"{(sc_spread*100):,.0f}" if not np.isnan(sc_spread) else "—",
              delta=f"{(sc_spread - latest_spread)*100:+.0f} vs base" if not any(map(np.isnan, [sc_spread, latest_spread])) else None)

with col4:
    st.metric("CAD/USD", f"{sc_fx:,.4f}" if not np.isnan(sc_fx) else "—",
              delta=f"{pct_change(fx, 21):+.2f}% (≈1m)" if not np.isnan(pct_change(fx, 21)) else None)

# Secondary KPIs row
col5, col6 = st.columns(2)
with col5:
    st.metric("VIX (level)", f"{sc_vix:,.2f}" if not np.isnan(sc_vix) else "—")
with col6:
    st.metric("US CPI YoY (%)", f"{latest_infl:,.2f}" if not np.isnan(latest_infl) else "—")

st.divider()

# =========================
# 8) Charts
# =========================
# Yield spread chart
if not yc_spread.empty:
    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(
        x=yc_spread.index, y=yc_spread.values, mode='lines', name='10Y-2Y Spread (%)'
    ))
    fig_spread.add_hline(y=0, line_dash="dot", opacity=0.5)
    fig_spread.update_layout(
        title="US Yield Curve Spread (10Y–2Y)",
        yaxis_title="Spread (%)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380
    )
    st.plotly_chart(fig_spread, use_container_width=True)
else:
    st.info("Yield spread unavailable (check data sources).")

# FX chart
if not fx.empty:
    fig_fx = go.Figure()
    fig_fx.add_trace(go.Scatter(x=fx.index, y=fx.values, mode='lines', name='CAD/USD'))
    fig_fx.update_layout(
        title="CAD/USD Exchange Rate",
        yaxis_title="CAD per USD (Close)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380
    )
    st.plotly_chart(fig_fx, use_container_width=True)
else:
    st.info("FX data unavailable.")

# CPI YoY chart
if not cpi_yoy.empty:
    fig_cpi = go.Figure()
    fig_cpi.add_trace(go.Scatter(x=cpi_yoy.index, y=cpi_yoy.values, mode='lines', name='CPI YoY %'))
    fig_cpi.update_layout(
        title="US CPI Year-over-Year (%)",
        yaxis_title="YoY Inflation (%)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380
    )
    st.plotly_chart(fig_cpi, use_container_width=True)
else:
    st.caption("US CPI YoY not shown (FRED unavailable).")

st.divider()

# =========================
# 9) Executive summary (auto-updating text box)
# =========================
summary_lines = []

if not np.isnan(sc_spread):
    curve_status = "inverted" if sc_spread < 0 else "normal (upward sloping)"
    summary_lines.append(f"- Yield curve is **{curve_status}**; current spread ≈ **{sc_spread*100:.0f} bps**.")
if not np.isnan(sc_fx):
    one_mo = pct_change(fx, 21)
    if not np.isnan(one_mo):
        dir_ = "weaker" if one_mo > 0 else "stronger"
        summary_lines.append(f"- CAD is **{abs(one_mo):.1f}% {dir_}** vs USD over ~1 month.")
if not np.isnan(sc_vix):
    vol_label = "elevated" if sc_vix >= 20 else "calm"
    summary_lines.append(f"- Equity volatility (VIX) is **{vol_label}** at **{sc_vix:.1f}**.")
if not np.isnan(latest_infl):
    summary_lines.append(f"- US YoY inflation prints **{latest_infl:.1f}%** (latest month).")

if summary_lines:
    st.markdown("### Executive Snapshot")
    st.markdown("\n".join(summary_lines))
else:
    st.caption("Executive snapshot will populate as data loads.")

# =========================
# 10) Footer
# =========================
st.write("")
st.caption("Data sources: FRED (if available), Yahoo Finance. This is a demo; scenario effects are illustrative.")
