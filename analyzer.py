# yield_curve_boc.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🇨🇦 Canada Yield Curve Analyzer", layout="wide")

# Bank of Canada series IDs for yields
BOND_SERIES = {
    "1-Year": "V39049",
    "2-Year": "V39050",
    "5-Year": "V39053",
    "10-Year": "V39056",
    "30-Year": "V39059"
}

@st.cache_data
def fetch_boc_data(series_id):
    """Fetch yield data from Bank of Canada CSV endpoint."""
    url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/csv"
    df = pd.read_csv(url, skiprows=10)  # skip metadata rows
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df

# Fetch latest yields
yields = {}
for label, sid in BOND_SERIES.items():
    df = fetch_boc_data(sid)
    latest_value = df.dropna().iloc[-1]["Value"]
    yields[label] = latest_value

# Display data
st.title("🇨🇦 Canada Government Bond Yield Curve")
st.write("Data source: Bank of Canada")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Latest Yields")
    st.dataframe(pd.DataFrame.from_dict(yields, orient="index", columns=["Yield (%)"]))

with col2:
    st.subheader("Yield Curve")
    fig, ax = plt.subplots()
    ax.plot(list(yields.keys()), list(yields.values()), marker="o")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Canada Yield Curve")
    ax.grid(True)
    st.pyplot(fig)

# Historical chart for a selected maturity
st.subheader("Historical Yield Trends")
maturity = st.selectbox("Select maturity:", list(BOND_SERIES.keys()))
hist_df = fetch_boc_data(BOND_SERIES[maturity])
st.line_chart(hist_df.set_index("Date")["Value"])


