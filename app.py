import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="iPoultry AI – Bird Weight Prediction",
                   layout="wide")

st.title("🐔 iPoultry AI Module – Bird Weight Forecast (XGBR Models)")


# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

weight_model = load_model("weight_xgb_model.pkl")
mort_model   = load_model("mortality_xgb_model.pkl")
fcr_model    = load_model("fcr_xgb_model.pkl")


# ----------------------------------------------------------
# IDEAL WEIGHT CHART (Ross)
# ----------------------------------------------------------
ideal_weight_chart = {
    0:0.043,1:0.061,2:0.079,3:0.099,4:0.122,5:0.148,6:0.176,7:0.208,8:0.242,9:0.28,
    10:0.321,11:0.366,12:0.414,13:0.465,14:0.519,15:0.576,16:0.637,17:0.701,18:0.768,
    19:0.837,20:0.91,21:0.985,22:1.062,23:1.142,24:1.225,25:1.309,26:1.395,27:1.483,
    28:1.573,29:1.664,30:1.757,31:1.851,32:1.946,33:2.041,34:2.138,35:2.235,
    36:2.332,37:2.43,38:2.527,39:2.625,40:2.723
}


# ----------------------------------------------------------
# INPUTS
# ----------------------------------------------------------
st.subheader("📥 Enter Today’s Farm Values")

col1, col2 = st.columns(2)

with col1:
    age_in_days = st.number_input("Age (Days)", 0, 100, 14, key="age_in_days")
    birds_alive = st.number_input("Birds Alive", 0, 200000, 900, key="birds_alive")
    mortality_today = st.number_input("Mortality Today", 0, 1000, 1, key="mortality_today")
    feed_today = st.number_input("Feed Today (kg)", 0.0, 2000.0, 22.0, key="feed_today")

with col2:
    water_today = st.number_input("Water (Liters)", 0.0, 5000.0, 30.0, key="water_today")
    sample_weight_today = st.number_input("Sample Bird Weight (kg)", 0.0, 5.0, 1.2, key="sample_weight_today")


# ----------------------------------------------------------
# RUN FORECAST
# ----------------------------------------------------------
if st.button("Predict Next 7 Days ▶️"):

    future_ages = np.arange(age_in_days, age_in_days + 8)  # today + next 7 days
    
    # Build feature set for each future day
    forecast_features = pd.DataFrame({
        "birds_alive": [birds_alive] * len(future_ages),
        "feed_kg": [feed_today] * len(future_ages),
        "water_consumption_l": [water_today] * len(future_ages),
        "mortality": [mortality_today] * len(future_ages),
        "age_in_days": future_ages
    })

    # Predictions using XGBR
    weight_preds = weight_model.predict(forecast_features)
    mort_preds   = mort_model.predict(forecast_features)
    fcr_preds    = fcr_model.predict(forecast_features)

    # Build output dataframe
    df_forecast = pd.DataFrame({
        "Bird Age (days)": future_ages,
        "Predicted Weight (kg)": np.round(weight_preds, 3),
        "Ideal Weight (kg)": [ideal_weight_chart.get(int(a), None) for a in future_ages],
        "Predicted Mortality (birds)": np.round(mort_preds).astype(int),
        "Predicted FCR": np.round(fcr_preds, 3)
    })

    # -----------------------------
    # Highlights for Day 33 & 35
    # -----------------------------
    day33 = df_forecast[df_forecast["Bird Age (days)"] == 33]
    day35 = df_forecast[df_forecast["Bird Age (days)"] == 35]

    st.markdown("## 📌 Key Highlights")

    if not day33.empty:
        st.markdown(
            f"""
            ### 🟦 **Day 33:**  
            **Predicted Weight:** {day33['Predicted Weight (kg)'].values[0]} kg  
            **Ideal Weight:** {day33['Ideal Weight (kg)'].values[0]} kg  
            """
        )

    if not day35.empty:
        st.markdown(
            f"""
            ### 🟩 **Day 35:**  
            **Predicted Weight:** {day35['Predicted Weight (kg)'].values[0]} kg  
            **Ideal Weight:** {day35['Ideal Weight (kg)'].values[0]} kg  
            """
        )


    # -----------------------------
    # SHOW TABLE
    # -----------------------------
    st.subheader("📊 Forecast Table")
    st.dataframe(
        df_forecast.style.format({
            "Predicted Weight (kg)": "{:.3f}",
            "Ideal Weight (kg)": "{:.3f}",
            "Predicted FCR": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------
    # PLOT
    # -----------------------------
    fig = go.Figure()

    # Prediction line
    fig.add_trace(go.Scatter(
        x=df_forecast["Bird Age (days)"],
        y=df_forecast["Predicted Weight (kg)"],
        mode="lines+markers",
        name="Predicted Weight (XGBR)",
        line=dict(width=3)
    ))

    # Ideal curve
    fig.add_trace(go.Scatter(
        x=df_forecast["Bird Age (days)"],
        y=df_forecast["Ideal Weight (kg)"],
        mode="lines+markers",
        name="Ideal Weight (Ross)",
        line=dict(width=3, dash="dash")
    ))

    fig.update_layout(
        title="Weight Curve: Predicted vs Ideal",
        xaxis_title="Age (Days)",
        yaxis_title="Weight (kg)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
