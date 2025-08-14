import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Canadian Yield Curve Analyzer", layout="wide")

# Mapping of maturities to Yahoo tickers
US_TREASURY_TICKERS = {
    "1M": "^IRX",
    "2Y": "^UST2Y",
    "5Y": "^UST5Y",
    "10Y": "^TNX",
    "30Y": "^TYX"
}


@st.cache_data(ttl=3600)
def fetch_yields():
    """Fetch latest yields from Yahoo Finance."""
    data = {}
    for maturity, ticker in CANADA_BOND_TICKERS.items():
        try:
            df = yf.download(ticker, period="1mo", interval="1d")
            if not df.empty:
                data[maturity] = df["Close"]
            else:
                st.warning(f"No data for {maturity} ({ticker})")
        except Exception as e:
            st.error(f"Error fetching {maturity}: {e}")
    return pd.DataFrame(data)

st.title("📈 Canadian Yield Curve Analyzer")
st.write("Live Canadian Government Bond Yields from Yahoo Finance.")

df = fetch_yields()

if df.empty:
    st.error("No data available.")
else:
    latest = df.iloc[-1].dropna()

    # Display table
    st.subheader("Latest Yields (%)")
    st.table(latest.apply(lambda x: round(x, 2)))

    # Plot yield curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=latest.index,
        y=latest.values,
        mode="lines+markers",
        line=dict(color="blue", width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="Canadian Yield Curve",
        xaxis_title="Maturity",
        yaxis_title="Yield (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)




