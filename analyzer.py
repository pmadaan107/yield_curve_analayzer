import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Canada Yield Curve", layout="centered")
st.title("🇨🇦 Canadian Government Bond Yield Curve")
st.write("Live data from Bank of Canada Valet API")

# Function to get all series metadata from BoC
@st.cache_data
def get_all_series():
    url = "https://www.bankofcanada.ca/valet/lists/series/json"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

# Function to find series IDs for given keywords
def find_series_ids(keywords):
    series_data = get_all_series()
    results = {}
    for key, name in keywords.items():
        for series_id, meta in series_data["seriesDetail"].items():
            if all(word.lower() in meta["label"].lower() for word in name.split()):
                results[key] = series_id
                break
    return results

# Function to fetch latest yields
@st.cache_data
def fetch_latest_yields(series_ids):
    yields = {}
    for label, sid in series_ids.items():
        url = f"https://www.bankofcanada.ca/valet/observations/{sid}/json"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        obs = data.get("observations", [])
        if obs:
            val = obs[-1][sid]["v"]
            yields[label] = float(val)
    return yields

# Keywords to search in series list
KEYWORDS = {
    "1Y": "1-year benchmark bond yield",
    "2Y": "2-year benchmark bond yield",
    "5Y": "5-year benchmark bond yield",
    "10Y": "10-year benchmark bond yield",
    "30Y": "30-year benchmark bond yield"
}

try:
    series_ids = find_series_ids(KEYWORDS)
    st.write("Series IDs found:", series_ids)

    if not series_ids:
        st.error("Could not find any matching series IDs from BoC API.")
    else:
        yields = fetch_latest_yields(series_ids)

        if yields:
            df = pd.DataFrame({
                "Term": list(yields.keys()),
                "Yield": list(yields.values())
            })

            st.subheader("Latest Yields (%)")
            st.dataframe(df)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Term"], y=df["Yield"], mode='lines+markers',
                name="Yield Curve"
            ))
            fig.update_layout(
                title="Canadian Government Bond Yield Curve",
                xaxis_title="Term to Maturity",
                yaxis_title="Yield (%)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No yield data returned from Bank of Canada API.")

except requests.exceptions.RequestException as e:
    st.error(f"Error fetching data: {e}")
