import streamlit as st
import pandas as pd
import numpy as np  # Import NumPy
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Load Data (Same as in the original script) ---
@st.cache_data # Use caching for efficiency
def load_data():
    car = pd.read_csv('processes2.csv')
    return car

car = load_data()


# --- Preprocessing (Simplified - Streamlit doesn't handle all preprocessing steps directly)---
def preprocess_data(df):
    df = df.drop(columns=['Unnamed: 0', 'Mileage Unit'])
    return df

processed_car = preprocess_data(car)

"""# Encoding Categorical Features"""
from sklearn.preprocessing import LabelEncoder, StandardScaler

for cat in categoric:
    encoder = LabelEncoder()

    car[cat] = encoder.fit_transform(car[cat])
    
"""# Standardize Numerical Features"""
for num in numeric:
    scaler = StandardScaler()
    car[num] = scaler.fit_transform(car[num])


# --- Model Building & Prediction (This is where the core logic goes)---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the best models from your original script (assuming they've been saved)
best_rfr = joblib.load('best_random_forest_regressor_model.joblib')
best_svr = joblib.load('best_SVR_Model.joblib')

def predict_price(df):
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    rfr = RandomForestRegressor()
    svr = SVR()

    # Fit models (you'd likely load pre-trained ones here)
    rfr.fit(X_train, y_train)
    svr.fit(X_train, y_train)


    y_pred_rfr = rfr.predict(X_test)
    y_pred_svr = svr.predict(X_test)

    y_pred_scaled_rfr = y_pred_rfr
    y_pred_scaled_svr = y_pred_svr

    # Get the mean and standard deviation for 'selling_price' from the fitted scaler
    # 'selling_price' is at index 1 in the 'numeric' columns
    # (year, selling_price, km_driven, seats, max_power (in bph), Mileage, Engine (CC))
    selling_price_col_idx = 1

    # Ensure that scaler has been fitted and has mean_ and scale_ attributes
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        selling_price_mean = scaler.mean_[selling_price_col_idx]
        selling_price_std = scaler.scale_[selling_price_col_idx]
    else:
        raise ValueError("Scaler has not been fitted, cannot inverse transform.")

    # Apply the inverse scaling formula to get original values
    y_pred_original_rfr = (y_pred_scaled_rfr * selling_price_std) + selling_price_mean
    y_pred_original_svr = (y_pred_scaled_svr * selling_price_std) + selling_price_mean
    #y_actual_original = (y_test.values * selling_price_std) + selling_price_mean

    return y_pred_original_rfr, y_pred_original_svr


# --- Streamlit UI ---
st.title('Car Price Prediction App')

# Sidebar for User Input
with st.sidebar:
    st.header("Input Car Features")
    feature_names = car.columns.tolist()  # Get column names dynamically
    selected_features = st.multiselect(
        'Select Features:',
        options=feature_names,
        default=['year', 'selling_price', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']) # Default selection

# Filter DataFrame based on user selections
filtered_car = car[selected_features]

# Prediction Section
if st.button('Predict Price'):
    y_pred_original_rfr, y_pred_original_svr = predict_price(filtered_car)

    st.subheader("Prediction Results:")
    st.write("Random Forest Regressor Predictions: ", y_pred_original_rfr)
    st.write("SVR Predictions: ", y_pred_original_svr)
    st.write("Average Car Price Predictions: ", y_pred_original_rfr+y_pred_original_svr/2)


# --- Display Data (Optional - for inspection)---
if st.checkbox('Show Raw Data'):
    st.subheader("Raw Car Data:")
    st.dataframe(car)
