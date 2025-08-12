import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('data/housing.csv')

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

# Home Page
if page == "Home":
    st.title("🏠 Boston Housing Price Prediction")
    st.write("""
    This app predicts **Boston Housing Prices** using a machine learning model.
    - Explore the dataset
    - View interactive visualizations
    - Make predictions with our trained model
    - Evaluate model performance
    """)

# Data Exploration Page
elif page == "Data Exploration":
    st.title("Data Exploration")
    
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:", df.dtypes.to_dict())
    
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    st.subheader("Data Description")
    st.dataframe(df.describe())
    
    st.subheader("Filter Data")
    column = st.selectbox("Select column to filter", df.columns)
    unique_values = df[column].unique()
    selected_value = st.selectbox(f"Select {column}", unique_values)
    filtered_df = df[df[column] == selected_value]
    st.dataframe(filtered_df)

# Visualizations Page
elif page == "Visualizations":
    st.title("Data Visualizations")
    
    st.subheader("Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['median_house_value'], kde=True, ax=ax1)
    st.pyplot(fig1)
    
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
    
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("X-axis", df.columns)
    y_axis = st.selectbox("Y-axis", df.columns)
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax3)
    st.pyplot(fig3)

# Model Prediction Page
elif page == "Model Prediction":
    st.title("Make a Prediction")
    
    st.write("Enter the following features to predict the house price:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input("Longitude", min_value=-124.35, max_value=-114.31, value=-119.57)
        latitude = st.number_input("Latitude", min_value=32.54, max_value=41.95, value=36.20)
        housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=52, value=28)
        total_rooms = st.number_input("Total Rooms", min_value=2, max_value=40000, value=2500)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=6500, value=500)
        
    with col2:
        population = st.number_input("Population", min_value=3, max_value=15000, value=1500)
        households = st.number_input("Households", min_value=1, max_value=6000, value=500)
        median_income = st.number_input("Median Income ($10,000s)", min_value=0.5, max_value=15.0, value=3.0)
        
        # Ocean proximity binary inputs
        st.write("Ocean Proximity Indicators:")
        inland = st.checkbox("INLAND")
        near_bay = st.checkbox("NEAR BAY")
        near_ocean = st.checkbox("NEAR OCEAN")
        island = st.checkbox("ISLAND")
        one_h_ocean = st.checkbox("1H OCEAN")
        # 1H OCEAN would be when none of these are selected

    # Calculate engineered features
    bedroom_ratio = total_bedrooms / total_rooms if total_rooms > 0 else 0
    household_rooms = total_rooms / households if households > 0 else 0

    # Make prediction
    if st.button("Predict Median House Value"):
        # Prepare input array in the exact same order as your model was trained on
        input_data = np.array([[
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            one_h_ocean,
            1 if inland else 0,
            1 if near_bay else 0,
            1 if near_ocean else 0,
            1 if island else 0,
            bedroom_ratio,
            household_rooms
        ]])
        
        prediction = model.predict(input_data)
        st.success(f"Predicted Median House Value: ${prediction[0]:,.2f}")

# Model Performance Page
