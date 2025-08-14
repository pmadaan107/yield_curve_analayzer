import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="🇨🇦 Canada Yield Curve", layout="centered")

st.title("🇨🇦 Canadian Government Bond Yield Curve")
st.write("Live data from Bank of Canada Valet API")

# Bank of Canada Valet API series IDs
SERIES = {
    "1Y": "V39055",
    "2Y": "V39056",
    "5Y": "V39057",
    "10Y": "V39058",
    "30Y": "V39062"
}

@st.cache_data
def fetch_latest_yields():
    yields = {}
    for label, sid in SERIES.items():
        url = f"https://www.bankofcanada.ca/valet/observations/{sid}/json"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        obs = data.get("observations", [])
        if obs:
            # Get latest observation
            latest = obs[-1]
            val = latest[sid]["v"]
            yields[label] = float(val)
    return yields

try:
    yields = fetch_latest_yields()

    if yields:
        # Create dataframe
        df = pd.DataFrame({
            "Term": list(yields.keys()),
            "Yield": list(yields.values())
        })

        st.subheader("Latest Yields (%)")
        st.dataframe(df)

        # Plot yield curve
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


