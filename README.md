# Boston Housing Price Predictor 🏠

## 🚀 Overview

This Streamlit web app predicts median house values in Boston using a machine learning regression model. It provides interactive data exploration, visualizations, real-time predictions, and model performance metrics.

## 📦 Features

- **Home:** Introduction and project overview.
- **Data Exploration:** Inspect the dataset, filter, and view statistics.
- **Visualizations:** Interactive charts (distribution, heatmap, scatter plots).
- **Model Prediction:** Input features and get instant price predictions.
- **Model Performance:** View metrics and plots for model evaluation.

## 📊 Dataset

Uses the Boston housing dataset (`data/housing.csv`) with features like location, median income, rooms, and proximity to the ocean.

## 🧠 Model

A pre-trained regression model (`model.pkl`) is used for predictions.

## 🛠️ Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/imashaRan12/ML-Model-Housing.git
   cd boston-housing-streamlit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🔧 Dependencies

- Python 3.7+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pillow

## 📁 Project Structure

```
boston-housing-streamlit/
│   app.py
│   model.pkl
│   README.md
│   requirements.txt
│
├── data/
│   ├── housing.csv
│   ├── X_test.csv
│   └── y_test.csv
└── notebooks/
    └── model_training.ipynb

```

## ✨ Credits

Developed by P.I.R.SENADHEERA
Dataset: [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
