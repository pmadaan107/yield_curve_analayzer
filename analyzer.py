import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# ---- CONFIG ----
st.set_page_config(page_title="Canadian Yield Curve (ETF Proxy)", layout="wide")
st.title("🇨🇦 Canadian Yield Curve (ETF Proxies)")

# ETF proxies for short, medium, and long-term bonds
ETF_TICKERS = {
    "1-3 Year": "ZFS.TO",
    "3-7 Year": "ZFM.TO",
    "10+ Year": "ZFL.TO"
}

@st.cache_data
def fetch_etf_yields():
    yields = []
    for term, ticker in ETF_TICKERS.items():
        try:
            etf = yf.Ticker(ticker)
            info = etf.info
            # 'yield' key is in decimal form, multiply by 100 for %
            dist_yield = info.get("yield", None)
            if dist_yield is not None:
                yields.append({"Term": term, "Yield (%)": dist_yield * 100})
            else:
                yields.append({"Term": term, "Yield (%)": None})
        except Exception as e:
            yields.append({"Term": term, "Yield (%)": None})
    return pd.DataFrame(yields)

# ---- FETCH DATA ----
df = fetch_etf_yields()

if df["Yield (%)"].isnull().all():
    st.error("No yield data available right now. Please try again later.")
else:
    # Plot
    fig = px.line(df, x="Term", y="Yield (%)", markers=True, title="Canadian Yield Curve (ETF Proxy)")
    fig.update_layout(yaxis_title="Yield (%)", xaxis_title="Term to Maturity")
    st.plotly_chart(fig, use_container_width=True)

    # Show table
    st.subheader("Yield Data")
    st.dataframe(df)

