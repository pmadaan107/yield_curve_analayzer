# yield_curve_analyzer.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime as dt

# -------------------
# SETTINGS
# -------------------
st.set_page_config(page_title="🇨🇦 Canadian Yield Curve Analyzer", layout="wide")
st.title("🇨🇦 Canadian Government Bond Yield Curve")

# Define Canadian yield codes from FRED
FRED_CODES = {
    "1 Year": "IRLTLT01CAM156N",  # Long-term rates (monthly) - will simulate 1Y for example
    "5 Year": "IRLTLT01CAM193N",  # Monthly
    "10 Year": "IRLTLT01CAM156N", # Reused as placeholder
    "30 Year": "IRLTLT01CAM156N"  # Reused as placeholder
}

# Date range selector
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=365)

# -------------------
# FETCH DATA
# -------------------
@st.cache_data
def get_fred_data(start, end):
    df = pd.DataFrame()
    for maturity, code in FRED_CODES.items():
        try:
            series = pdr.DataReader(code, "fred", start, end)
            df[maturity] = series
        except Exception as e:
            st.warning(f"Could not fetch {maturity} data: {e}")
    return df

df = get_fred_data(start_date, end_date)

# -------------------
# DISPLAY
# -------------------
st.subheader("Yield Data (Latest)")
st.dataframe(df.tail(10))

st.subheader("Yield Curve Plot")
fig, ax = plt.subplots()
df.plot(ax=ax)
ax.set_title("Canadian Government Bond Yields")
ax.set_ylabel("Yield (%)")
ax.set_xlabel("Date")
st.pyplot(fig)

# Show latest curve snapshot
st.subheader("Latest Yield Curve Snapshot")
latest = df.iloc[-1].dropna()
fig2, ax2 = plt.subplots()
ax2.plot(latest.index, latest.values, marker='o')
ax2.set_title(f"Yield Curve as of {df.index[-1].date()}")
ax2.set_ylabel("Yield (%)")
ax2.set_xlabel("Maturity")
st.pyplot(fig2)
