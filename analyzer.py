# yield_curve_investing.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import investpy

st.set_page_config(page_title="🇨🇦 Canada Yield Curve Analyzer", layout="wide")

# Define maturities and investing.com search names
BOND_SEARCH = {
    "1-Year": "Canada 1-Year",
    "2-Year": "Canada 2-Year",
    "5-Year": "Canada 5-Year",
    "10-Year": "Canada 10-Year",
    "30-Year": "Canada 30-Year"
}

@st.cache_data
def fetch_investing_yields():
    yields = {}
    for label, search_name in BOND_SEARCH.items():
        try:
            df = investpy.get_bond_historical_data(
                bond=search_name,
                from_date="01/01/2024",
                to_date=pd.Timestamp.today().strftime("%d/%m/%Y")
            )
            latest_value = df["Close"].iloc[-1]
            yields[label] = latest_value
        except Exception as e:
            yields[label] = None
    return yields

yields = fetch_investing_yields()

st.title("🇨🇦 Canada Government Bond Yield Curve")
st.write("Data source: Investing.com via `investpy`")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Latest Yields")
    st.dataframe(pd.DataFrame.from_dict(yields, orient="index", columns=["Yield (%)"]))

with col2:
    st.subheader("Yield Curve")
    fig, ax = plt.subplots()
    labels = list(yields.keys())
    values = [yields[k] for k in labels]
    ax.plot(labels, values, marker="o")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Canada Yield Curve")
    ax.grid(True)
    st.pyplot(fig)

