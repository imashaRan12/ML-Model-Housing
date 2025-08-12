import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #2A7B9B, #57C785);
        color: #222222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {
        background-color: #283845;
        color: #f0f0f0;
    }
    .title-style {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    .subtitle-style {
        color: #34495e;
        font-style: italic;
        text-align: center;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    div.stButton > button {
        background-color: #1abc9c;
        color: white;
        border-radius: 6px;
        height: 3em;
        font-size: 1rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #16a085;
        cursor: pointer;
    }
    .block-container {
        padding: 3rem 5rem;
    }
    </style>
    """, unsafe_allow_html=True
)

# Page config for wider layout and nicer title
st.set_page_config(
    page_title="Boston Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Sidebar navigation with descriptions
st.sidebar.title("☰ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]
)

# --- Home Page ---
if page == "Home":
    st.markdown('<h1 class="title-style">🏠 Boston Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-style">A modern web app to predict housing prices using machine learning.</p>', unsafe_allow_html=True)
    
    st.write("""
    Welcome to the Boston Housing Price Prediction app! This project leverages **machine learning** to analyze housing data and predict median house values.
    """)

    with st.expander("ℹ️ About this project"):
        st.write("""
        This app provides:
        - **Data Exploration:** Understand the dataset and its characteristics.
        - **Visualizations:** Explore interactive charts and correlations.
        - **Prediction:** Input your own features and get price predictions in real time.
        - **Model Performance:** Evaluate the accuracy of our trained model with standard metrics.
        
        The dataset used is from the Boston housing dataset, featuring key housing attributes such as location, median income, and number of rooms.
        
        The model is a pre-trained regression model saved as a pickle file, providing quick and accurate predictions.
        """)

    st.write("---")
    st.markdown("### 🚀 Quick Navigation")
    st.markdown("""
    Use the sidebar to navigate between:
    - **Data Exploration** to peek into the raw data,
    - **Visualizations** for insightful charts,
    - **Model Prediction** to test your own inputs,
    - **Model Performance** to check how well the model performs.
    """)

    # Add a nice image or illustration if you have one:
    try:
        image = Image.open('assets/boston_house.jpg')  # Place an image in your assets folder
        st.image(image, caption="Boston Housing Dataset Overview", use_column_width=True)
    except FileNotFoundError:
        pass

# --- Data Exploration Page ---
elif page == "Data Exploration":
    st.markdown('<h1 class="title-style">🔎 Data Exploration</h1>', unsafe_allow_html=True)

    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:", df.dtypes.to_dict())
    
    st.subheader("Sample Data")
    st.dataframe(df.sample(10))  # Show a random sample for more variety
    
    st.subheader("Data Description")
    st.dataframe(df.describe())
    
    st.subheader("Filter Data")
    column = st.selectbox("Select column to filter", df.columns)
    unique_values = df[column].unique()
    
    # If many unique values, provide a search-enabled selectbox or multiselect
    if len(unique_values) > 30:
        selected_values = st.multiselect(f"Select {column} (multiple allowed)", unique_values[:100])
        if selected_values:
            filtered_df = df[df[column].isin(selected_values)]
        else:
            filtered_df = df
    else:
        selected_value = st.selectbox(f"Select {column}", unique_values)
        filtered_df = df[df[column] == selected_value]
    st.dataframe(filtered_df)

# --- Visualizations Page ---
elif page == "Visualizations":
    st.markdown('<h1 class="title-style">📊 Data Visualizations</h1>', unsafe_allow_html=True)
    
    st.subheader("Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['median_house_value'], kde=True, ax=ax1, color="#3498db")
    ax1.set_xlabel("Median House Value")
    st.pyplot(fig1)
    
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2, fmt=".2f")
    st.pyplot(fig2)
    
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis", df.columns, index=df.columns.get_loc("median_income"))
    y_axis = st.selectbox("Select Y-axis", df.columns, index=df.columns.get_loc("median_house_value"))
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax3, alpha=0.6)
    st.pyplot(fig3)

# --- Model Prediction Page ---
elif page == "Model Prediction":
    st.markdown('<h1 class="title-style">🏡 Make a Prediction</h1>', unsafe_allow_html=True)
    
    st.write("Fill in the housing features below to predict the median house value:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input("Longitude", min_value=-124.35, max_value=-114.31, value=-119.57, format="%.5f")
        latitude = st.number_input("Latitude", min_value=32.54, max_value=41.95, value=36.20, format="%.5f")
        housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=52, value=28)
        total_rooms = st.number_input("Total Rooms", min_value=2, max_value=40000, value=2500)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=6500, value=500)
        
    with col2:
        population = st.number_input("Population", min_value=3, max_value=15000, value=1500)
        households = st.number_input("Households", min_value=1, max_value=6000, value=500)
        median_income = st.number_input("Median Income ($10,000s)", min_value=0.5, max_value=15.0, value=3.0, format="%.2f")
        
        st.write("### Ocean Proximity Indicators (select one):")
        proximity_options = ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND", "1H OCEAN"]
        selected_proximity = st.radio("Select Ocean Proximity", proximity_options, index=0)
        
    # Create binary encoding for proximity
    proximity_encoding = {
        "INLAND": [1,0,0,0,0],
        "NEAR BAY": [0,1,0,0,0],
        "NEAR OCEAN": [0,0,1,0,0],
        "ISLAND": [0,0,0,1,0],
        "1H OCEAN": [0,0,0,0,1]
    }
    
    one_h_ocean, inland, near_bay, near_ocean, island = proximity_encoding[selected_proximity]

    # Calculate engineered features
    bedroom_ratio = total_bedrooms / total_rooms if total_rooms > 0 else 0
    household_rooms = total_rooms / households if households > 0 else 0

    # Prediction button
    if st.button("Predict Median House Value"):
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
            inland,
            near_bay,
            near_ocean,
            island,
            bedroom_ratio,
            household_rooms
        ]])
        
        try:
            prediction = model.predict(input_data)
            st.success(f"🏠 Predicted Median House Value: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- Model Performance Page ---
elif page == "Model Performance":
    st.markdown('<h1 class="title-style">📈 Model Performance</h1>', unsafe_allow_html=True)

    # Load test data
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation Metrics")
    st.markdown(f"""
    - **Mean Absolute Error (MAE):** {mae:.2f}  
    - **Mean Squared Error (MSE):** {mse:.2f}  
    - **Root Mean Squared Error (RMSE):** {rmse:.2f}  
    - **R² Score:** {r2:.2f}  
    """)

    # Scatter plot: Predicted vs Actual
    st.subheader("Predicted vs Actual Values")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_pred, alpha=0.6, color="#1f77b4")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual")
    st.pyplot(fig)

    # Residuals distribution
    st.subheader("Residuals Distribution")
    residuals = y_test.values.flatten() - y_pred
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(residuals, bins=30, color="#9467bd", alpha=0.7)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)
