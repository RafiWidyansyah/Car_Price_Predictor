import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Load Data ---
@st.cache_data 
def load_data():
    car = pd.read_csv('processes2.csv')
    return car

car = load_data()

# --- Define Feature Groups (Needed for the loops below) ---
# We define these based on your dataset structure
numeric = ['year', 'selling_price', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']
categoric = [col for col in car.columns if col not in numeric and col not in ['Unnamed: 0', 'Mileage Unit']]

# --- Preprocessing ---
def preprocess_data(df):
    # Drop columns only if they exist to avoid errors
    cols_to_drop = [c for c in ['Unnamed: 0', 'Mileage Unit'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df

car = preprocess_data(car)

# Encoding Categorical Features
for cat in categoric:
    if cat in car.columns:
        encoder = LabelEncoder()
        car[cat] = encoder.fit_transform(car[cat].astype(str))
    
# Standardize Numerical Features
scaler = StandardScaler()
# We only scale columns that actually exist in the dataframe
existing_numeric = [col for col in numeric if col in car.columns]
car[existing_numeric] = scaler.fit_transform(car[existing_numeric])

# --- Model Loading ---
# We load these once at the start using cache_resource
@st.cache_resource
def load_models():
    rfr = joblib.load('best_random_forest_regressor_model.joblib')
    svr = joblib.load('best_SVR_Model.joblib')
    return rfr, svr

best_rfr, best_svr = load_models()

# --- Prediction Function ---
def predict_price(df):
    # FIX: Check if selling_price exists before dropping to avoid KeyError
    if 'selling_price' in df.columns:
        X = df.drop('selling_price', axis=1)
    else:
        X = df

    # Use the pre-loaded "best" models for prediction
    y_pred_rfr = best_rfr.predict(X)
    y_pred_svr = best_svr.predict(X)

    # Get scaling parameters for inverse transform
    # 'selling_price' index in the original scaler depends on its position in 'existing_numeric'
    try:
        selling_price_col_idx = existing_numeric.index('selling_price')
        selling_price_mean = scaler.mean_[selling_price_col_idx]
        selling_price_std = scaler.scale_[selling_price_col_idx]
    except (ValueError, IndexError):
        selling_price_mean, selling_price_std = 0, 1

    # Apply the inverse scaling formula
    y_pred_original_rfr = (y_pred_rfr * selling_price_std) + selling_price_mean
    y_pred_original_svr = (y_pred_svr * selling_price_std) + selling_price_mean

    return y_pred_original_rfr, y_pred_original_svr

# --- Streamlit UI ---
st.title('Car Price Prediction App')

# Sidebar for User Input
with st.sidebar:
    st.header("Input Car Features")
    feature_names = car.columns.tolist() 
    selected_features = st.multiselect(
        'Select Features:',
        options=feature_names,
        default=['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)'])

# Filter DataFrame based on user selections
filtered_car = car[selected_features]

# Prediction Section
if st.button('Predict Price'):
    if filtered_car.empty:
        st.error("Please select at least one feature.")
    else:
        y_pred_original_rfr, y_pred_original_svr = predict_price(filtered_car)

        st.subheader("Prediction Results:")
        # Displaying the first prediction if multiple rows are passed
        st.write(f"**Random Forest Regressor:** {y_pred_original_rfr[0]:,.2f}")
        st.write(f"**SVR Predictions:** {y_pred_original_svr[0]:,.2f}")
        
        # Fixed math for average: (A + B) / 2
        avg_price = (y_pred_original_rfr[0] + y_pred_original_svr[0]) / 2
        st.write(f"**Average Car Price Prediction:** {avg_price:,.2f}")

# --- Display Data ---
if st.checkbox('Show Raw Data'):
    st.subheader("Raw Car Data:")
    st.dataframe(car)
