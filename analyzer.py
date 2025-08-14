# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import datetime as dt

st.set_page_config(page_title="Yield Curve Analyzer (Yahoo Finance)", layout="wide")
st.title("Yield Curve Analyzer (via Yahoo Finance)")
st.caption("Demo using U.S. Treasury yields — replace with Canadian equivalents if available.")

# Maturity tickers for US Treasuries (replace with Canadian tickers if found)
TICKERS = {
    "1M": "^IRX",   # 13-week T-bill (proxy for short end)
    "2Y": "^TNX2Y",
    "5Y": "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX"
}

@st.cache_data(ttl=3600)
def fetch_yields():
    data = {}
    for label, ticker in TICKERS.items():
        df = yf.download(ticker, period="6mo", interval="1d")
        if not df.empty:
            data[label] = df['Close']
        else:
            st.warning(f"No data for ticker: {ticker}")
    if not data:
        st.error("No yield data found. Please check tickers.")
        return pd.DataFrame()  # return empty safely
    return pd.DataFrame(data)


df = fetch_yields().dropna()

# Sidebar
min_date = df.index.min().date()
max_date = df.index.max().date()
mode = st.sidebar.radio("Compare", ["Two dates", "Two ranges"], index=0)

if mode == "Two dates":
    dA = st.sidebar.date_input("Date A", value=max_date - dt.timedelta(days=30), min_value=min_date, max_value=max_date)
    dB = st.sidebar.date_input("Date B", value=max_date, min_value=min_date, max_value=max_date)
    curveA = df.loc[df.index.get_loc(pd.Timestamp(dA), method="nearest")]
    curveB = df.loc[df.index.get_loc(pd.Timestamp(dB), method="nearest")]
    labelA, labelB = str(dA), str(dB)
else:
    window_days = st.sidebar.slider("Window length (days)", 5, 90, 30)
    endA = st.sidebar.date_input("Window A end", value=max_date - dt.timedelta(days=30))
    endB = st.sidebar.date_input("Window B end", value=max_date)
    startA = endA - dt.timedelta(days=window_days)
    startB = endB - dt.timedelta(days=window_days)
    curveA = df.loc[str(startA):str(endA)].mean()
    curveB = df.loc[str(startB):str(endB)].mean()
    labelA, labelB = f"{startA}→{endA}", f"{startB}→{endB}"

terms = [float(t.replace("Y","").replace("M","0.083")) if "M" in t else float(t.replace("Y","")) for t in TICKERS.keys()]
yA, yB = curveA.values, curveB.values

# Plot curves
fig = go.Figure()
fig.add_trace(go.Scatter(x=terms, y=yA, mode="lines+markers", name=f"{labelA}"))
fig.add_trace(go.Scatter(x=terms, y=yB, mode="lines+markers", name=f"{labelB}"))
fig.update_layout(title="Yield Curves", xaxis_title="Maturity (Years)", yaxis_title="Yield (%)")
st.plotly_chart(fig, use_container_width=True)

# Insights (2Y–10Y slope)
slopeA = np.interp(10, terms, yA) - np.interp(2, terms, yA)
slopeB = np.interp(10, terms, yB) - np.interp(2, terms, yB)
st.write(f"2Y–10Y slope change: {slopeB - slopeA:.2f} pp")


