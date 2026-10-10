import streamlit as st
import pandas as pd
import joblib

# Load model & columns
model = joblib.load("xgboost_model.pkl")
columns = joblib.load("columns.pkl")

# Page config
st.set_page_config(page_title="Sales Prediction", layout="centered")

# 💙 CUSTOM CSS (BLUE THEME)
st.markdown("""
<style>
body {
    background-color: #f0f6ff;
}
.main {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}
h1 {
    color: #1565c0;
    text-align: center;
}
h4 {
    text-align: center;
    color: #0d47a1;
}
.stButton>button {
    background-color: #1565c0;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #0d47a1;
}
.sidebar .sidebar-content {
    background-color: #e3f2fd;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>📊 Inventory Demand Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<h4>Predict Weekly Sales using Machine Learning</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar inputs
st.sidebar.header("📥 Input Details")

store = st.sidebar.number_input("Store", min_value=1, value=1)
dept = st.sidebar.number_input("Department", min_value=1, value=1)

# Date input
date = st.sidebar.date_input("Select Date")

year = date.year
month = date.month
week = date.isocalendar()[1]
day = date.day
quarter = (month - 1)//3 + 1

# Holiday dropdown
is_holiday = st.sidebar.selectbox("Is Holiday?", ["No", "Yes"])
is_holiday = 1 if is_holiday == "Yes" else 0

# Section 1
st.markdown("### 🔵 Economic Indicators")
temperature = st.number_input("Temperature (scaled)", value=0.0)
fuel_price = st.number_input("Fuel Price (scaled)", value=0.0)
cpi = st.number_input("CPI (scaled)", value=0.0)
unemployment = st.number_input("Unemployment (scaled)", value=0.0)
size = st.number_input("Store Size (scaled)", value=0.0)

# Section 2
st.markdown("### 🟦 Sales History")
last_week_sales = st.number_input("Last Week Sales", value=20000.0)
rolling_sales = st.number_input("Rolling 4 Week Sales", value=18000.0)

st.markdown("---")

# Predict button
if st.button("🚀 Predict Sales"):

    input_data = pd.DataFrame([{
        'Store': store,
        'Dept': dept,
        'IsHoliday': is_holiday,

        'Temperature': temperature,
        'Fuel_Price': fuel_price,
        'CPI': cpi,
        'Unemployment': unemployment,
        'Size': size,

        'MarkDown1': 0,
        'MarkDown2': 0,
        'MarkDown3': 0,
        'MarkDown4': 0,
        'MarkDown5': 0,

        'Year': year,
        'Month': month,
        'Week': week,
        'Day': day,
        'Quarter': quarter,

        'Type_A': 1,
        'Type_B': 0,
        'Type_C': 0,

        'Last_Week_Sales': last_week_sales,
        'Rolling_4_Week_Sales': rolling_sales
    }])

    input_data = input_data[columns]

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Weekly Sales: ₹ {prediction[0]:,.2f}")

# Footer
st.markdown("---")
st.caption("⚠️ Inputs are standardized values used during training.")