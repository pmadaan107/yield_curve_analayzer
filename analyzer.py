import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
FRED_API_KEY = "YOUR_FRED_API_KEY"  # Get it free from https://fred.stlouisfed.org/
CANADA_YIELD_SERIES = {
    "1Y": "IRLTLT01CAM156N",  # Canada 1-Year Government Bond Yield
    "2Y": "IRLTLT02CAM156N",  # Canada 2-Year
    "5Y": "IRLTLT05CAM156N",  # Canada 5-Year
    "10Y": "IRLTLT10CAM156N", # Canada 10-Year
    "30Y": "IRLTLT30CAM156N"  # Canada 30-Year
}

# =========================
# FUNCTIONS
# =========================
@st.cache_data
def fetch_fred_data(series_id):
    """Fetch a single FRED series as a DataFrame."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("observations", [])
    df = pd.DataFrame(data)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_latest_yields():
    """Get the latest yield for each maturity."""
    latest_data = {}
    for label, series in CANADA_YIELD_SERIES.items():
        df = fetch_fred_data(series)
        latest_row = df.dropna().iloc[-1]
        latest_data[label] = latest_row["value"]
    return latest_data

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="🇨🇦 Canada Yield Curve Analyzer", layout="wide")
st.title("🇨🇦 Canada Yield Curve Analyzer")

st.markdown("""
This app fetches **Canadian Government Bond Yields** directly from the FRED API and plots the yield curve.
Data source: [FRED](https://fred.stlouisfed.org/)  
""")

try:
    latest_yields = get_latest_yields()
    st.subheader("📊 Latest Yields (%)")
    st.write(pd.DataFrame(latest_yields, index=["Yield (%)"]).T)

    # Plot yield curve
    fig, ax = plt.subplots()
    maturities = list(latest_yields.keys())
    yields = list(latest_yields.values())
    ax.plot(maturities, yields, marker="o", linestyle="-", color="b")
    ax.set_title("Canada Yield Curve")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.grid(True)

    st.pyplot(fig)

except requests.exceptions.RequestException as e:
    st.error(f"Error fetching data: {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")

