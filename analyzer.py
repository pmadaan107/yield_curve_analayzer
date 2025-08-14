import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Canadian Yield Curve", layout="wide")
st.title("🇨🇦 Canadian Government Bond Yield Curve")

# Try known possible series IDs from BoC
BOND_SERIES = {
    "1 Year": "V80691343",
    "2 Year": "V80691366",
    "5 Year": "V80691390",
    "10 Year": "V80691414",
    "30 Year": "V80691438"
}

def fetch_boc_series(series_id):
    """Fetch latest yield data for a given BoC Valet series if it exists."""
    url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/json"
    r = requests.get(url)
    if r.status_code != 200:
        st.warning(f"No data for series {series_id}")
        return None
    data = r.json()
    obs = pd.DataFrame(data["observations"])
    obs["date"] = pd.to_datetime(obs["d"])
    obs[series_id] = pd.to_numeric(obs[series_id].apply(lambda x: x.get("v", None)), errors="coerce")
    return obs[["date", series_id]]

# Gather yields
yields = {}
latest_date = None

for label, sid in BOND_SERIES.items():
    df = fetch_boc_series(sid)
    if df is not None and not df.dropna().empty:
        latest_val = df.dropna().iloc[-1]
        yields[label] = latest_val[sid]
        if latest_date is None:
            latest_date = latest_val["date"]

if yields:
    df_curve = pd.DataFrame({
        "Term": list(yields.keys()),
        "Yield (%)": list(yields.values())
    })
    st.write(f"**Latest Data Date:** {latest_date.date()}")
    st.table(df_curve)
    fig = px.line(df_curve, x="Term", y="Yield (%)", markers=True,
                  title="Canadian Government Bond Yield Curve")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No yield data found from BoC API.")
