import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from fredapi import Fred

# ====== API Keys ======

import os
from fredapi import Fred

FRED_KEY = os.getenv("FRED_API_KEY")

def fred_client() -> Fred | None:
    if not FRED_KEY:
        return None
    try:
        fred = Fred(api_key=FRED_KEY)
        # Lightweight ping to fail fast if key is bad
        _ = fred.get_series('DGS10', observation_start='2024-01-01')
        return fred
    except Exception as e:
        print(f"[FRED init error] {e}")  # you’ll see the exact API message
        return None

fred = fred_client()


# ====== 1. Yield Curve Example (US) ======
ten_year = fred.get_series('DGS10')  # 10-year Treasury
two_year = fred.get_series('DGS2')   # 2-year Treasury
yield_curve_df = pd.DataFrame({'10Y': ten_year, '2Y': two_year})
yield_curve_df['Spread'] = yield_curve_df['10Y'] - yield_curve_df['2Y']

# ====== 2. FX Example ======
fx_data = yf.download('CADUSD=X', period='1y', interval='1d')['Close']

# ====== 3. Plot Yield Curve Spread ======
fig = go.Figure()
fig.add_trace(go.Scatter(x=yield_curve_df.index, y=yield_curve_df['Spread'], mode='lines', name='10Y-2Y Spread'))
fig.update_layout(title='US Yield Curve Spread (10Y-2Y)', yaxis_title='Spread (%)')
fig.show()
import streamlit as st

st.set_page_config(page_title="🌍 Global Risk Dashboard", layout="wide")
st.title("Global Risk Dashboard")

# Sidebar Filters
country = st.sidebar.selectbox("Select Country", ["US", "Canada", "EU"])
metric = st.sidebar.multiselect("Select Metrics", ["Yield Curve", "FX", "Inflation", "Commodities"], default=["Yield Curve", "FX"])

# Display charts dynamically
if "Yield Curve" in metric:
    st.plotly_chart(fig, use_container_width=True)

if "FX" in metric:
    st.line_chart(fx_data)
