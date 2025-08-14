# yield_curve_canada.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Canadian Yield Curve", layout="wide")
st.title("🇨🇦 Canadian Government Bond Yield Curve")

# Bank of Canada Valet Series IDs
BOND_SERIES = {
    "1 Year": "V80691343",
    "2 Year": "V80691366",
    "5 Year": "V80691390",
    "10 Year": "V80691414",
    "30 Year": "V80691438"
}

def fetch_boc_series(series_id):
    """Fetch latest yield data for a given BoC Valet series."""
    url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/json"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    obs = pd.DataFrame(data["observations"])
    obs["date"] = pd.to_datetime(obs["d"])
    obs[series_id] = pd.to_numeric(obs[series_id].apply(lambda x: x.get("v", None)), errors="coerce")
    return obs[["date", series_id]]

# Fetch latest yields
yields = {}
latest_date = None

for label, sid in BOND_SERIES.items():
    df = fetch_boc_series(sid)
    latest_val = df.dropna().iloc[-1]
    yields[label] = latest_val[sid]
    if latest_date is None:
        latest_date = latest_val["date"]

# Create DataFrame for plotting
df_curve = pd.DataFrame({
    "Term": list(yields.keys()),
    "Yield (%)": list(yields.values())
})

# Display latest date and table
st.write(f"**Latest Data Date:** {latest_date.date()}")
st.table(df_curve)

# Plot yield curve
fig = px.line(
    df_curve,
    x="Term",
    y="Yield (%)",
    markers=True,
    title="Canadian Government Bond Yield Curve"
)
st.plotly_chart(fig, use_container_width=True)
