import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(page_title="Home Rent Predictor", layout="wide")

st.title("🏠 Home Rent Prediction Dashboard")
st.write("This app uses a **Linear Regression** model to predict rent based on property area.")

# 1. Load Data
try:
    d = pd.read_csv('home rent (1).csv')
except FileNotFoundError:
    # Creating sample data if the file isn't uploaded yet
    data = {
        'area': [2600, 3000, 3200, 3600, 4000],
        'price': [550000, 565000, 610000, 680000, 725000]
    }
    d = pd.DataFrame(data)

# 2. Train Model
model = LinearRegression()
model.fit(d[['area']], d['price'])

# --- SIDEBAR: User Input ---
st.sidebar.header("Prediction Input")
input_area = st.sidebar.number_input("Enter Property Area (sq ft):", min_value=500, max_value=10000, value=3000, step=100)

# 3. Perform Prediction
predicted_price = model.predict([[input_area]])

# --- MAIN PAGE: Results ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(d, use_container_width=True)
    
    st.metric(label="Predicted Rent Price", value=f"${predicted_price[0]:,.2f}")
    st.info(f"Model Formula: Price = ({model.coef_[0]:.2f} * Area) + {model.intercept_:.2f}")

with col2:
    st.subheader("Regression Visualization")
    fig, ax = plt.subplots()
    
    # Scatter plot of original data
    ax.scatter(d['area'], d['price'], color='red', label='Actual Data')
    
    # Regression line
    x_range = np.linspace(d['area'].min(), d['area'].max(), 100).reshape(-1, 1)
    ax.plot(x_range, model.predict(x_range), color='blue', label='Regression Line')
    
    # Highlight the current prediction
    ax.scatter(input_area, predicted_price, color='green', s=100, label='Your Prediction')
    
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
