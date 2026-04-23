import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Load Data & Models ---
@st.cache_data
def load_data():
    # Loading the original data to get categories/scaling right
    return pd.read_csv('processes2.csv')

@st.cache_resource
def load_models():
    # Load your pre-trained models
    rfr = joblib.load('best_random_forest_regressor_model.joblib')
    svr = joblib.load('best_SVR_Model.joblib')
    return rfr, svr

car = load_data()
best_rfr, best_svr = load_models()

# --- 2. Simplified Preprocessing ---
# Note: In a real app, you'd want to use the same Scaler 
# you used during training, not create a new one here.
scaler = StandardScaler()
# Assuming 'selling_price' is what we need to inverse scale later
# We fit it here just to simulate your existing logic
scaler.fit(car[['selling_price']]) 

# --- 3. Prediction Function ---
def predict_price(input_data):
    # We no longer "drop" selling_price here because the 
    # input_data should ONLY contain features.
    
    y_pred_rfr = best_rfr.predict(input_data)
    y_pred_svr = best_svr.predict(input_data)

    # Use the mean/std from the scaler for 'selling_price'
    selling_price_mean = scaler.mean_[0]
    selling_price_std = np.sqrt(scaler.var_[0])

    # Inverse scaling
    y_pred_original_rfr = (y_pred_rfr * selling_price_std) + selling_price_mean
    y_pred_original_svr = (y_pred_svr * selling_price_std) + selling_price_mean

    return y_pred_original_rfr, y_pred_original_svr

# --- 4. Streamlit UI ---
st.title('🚗 Car Price Prediction App')

with st.sidebar:
    st.header("Input Car Features")
    # Remove 'selling_price' from selectable features for prediction
    all_features = [col for col in car.columns if col not in ['selling_price', 'Unnamed: 0', 'Mileage Unit']]
    
    selected_features = st.multiselect(
        'Select Features for Model:',
        options=all_features,
        default=['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)'])

# Create a filtered dataframe for the model
# In a real scenario, you'd use st.number_input or st.selectbox 
# to let users input NEW values.
filtered_input = car[selected_features].head(1) # Taking 1 row as an example

if st.button('Predict Price'):
    try:
        res_rfr, res_svr = predict_price(filtered_input)

        st.subheader("Prediction Results:")
        col1, col2 = st.columns(2)
        col1.metric("Random Forest", f"${res_rfr[0]:,.2f}")
        col2.metric("SVR", f"${res_svr[0]:,.2f}")
        
        avg = (res_rfr[0] + res_svr[0]) / 2
        st.write(f"**Average Predicted Price:** ${avg:,.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

if st.checkbox('Show Raw Data'):
    st.dataframe(car)
