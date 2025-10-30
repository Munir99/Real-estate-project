import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Portland Housing Market Forecast", layout="wide")

st.title("🏠 Portland Housing Market Forecast")
st.markdown("""
This dashboard summarizes trends, seasonality, and forecasts in Portland’s real estate market
using historical home sale prices.
""")

# --- Load data ---
@st.cache_data
def load_data():
    # Adjust filenames if they differ in your folder
    monthly = pd.read_csv("monthly_avg_prices.csv")
    forecast = pd.read_csv("forecast_results.csv")
    return monthly, forecast

try:
    monthly_df, forecast_df = load_data()
except FileNotFoundError:
    st.error("Missing CSV files. Please make sure 'monthly_avg_prices.csv' and 'forecast_results.csv' are in the same folder as this app.")
    st.stop()

# --- Historical Trend ---
st.subheader("📈 Historical Monthly Average Prices")
fig1 = px.line(monthly_df, x="Sold Date", y="Average Price",
               title="Monthly Average Sold Prices in Portland",
               labels={"Sold Date": "Date", "Average Price": "Price (USD)"},
               markers=True)
st.plotly_chart(fig1, use_container_width=True)

# --- Forecast Section ---
st.subheader("🔮 12-Month Forecast (SARIMA Model)")
fig2 = px.line(forecast_df, x="Date", y="Forecast",
               title="Next 12-Month Forecast of Average Prices",
               labels={"Date": "Date", "Forecast": "Predicted Price (USD)"},
               markers=True, color_discrete_sequence=["red"])
st.plotly_chart(fig2, use_container_width=True)

# --- Findings Summary ---
st.subheader("🧭 Key Findings")
st.markdown("""
- **Market Direction:** Prices trended downward in early 2023 but began recovering toward 2025.  
- **Volatility:** Seasonal dips align with winter months, with consistent rebounds in spring.  
- **Forecast Insight:** SARIMA predicts moderate but steady appreciation in the next 12 months.  
- **Practical Takeaway:** Useful for sellers considering optimal listing periods or buyers watching for timing advantages.
""")

st.info("This app uses results from pre-trained models; no training occurs on Render. All forecasts are based on finalized analysis from the Jupyter notebook.")
