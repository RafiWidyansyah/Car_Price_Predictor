import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Data & Models ---
@st.cache_data
def load_data():
    return pd.read_csv('processes2.csv')

@st.cache_resource
def load_models():
    try:
        rfr = joblib.load('best_random_forest_regressor_model.joblib')
        svr = joblib.load('best_SVR_Model.joblib')
        return rfr, svr
    except:
        return None, None

car = load_data()
best_rfr, best_svr = load_models()

# --- 2. Sidebar Setup for User Input ---
st.sidebar.header("📝 Enter Car Specifications")

def user_input_features():
    # Numerical Features
    year = st.sidebar.slider('Vehicle Year', 2000, 2024, 2015)
    km_driven = st.sidebar.number_input('Kilometer (KM Driven)', value=0)
    seats = st.sidebar.selectbox('Number of Seats', [2, 4, 5, 7, 8])
    max_power = st.sidebar.number_input('Max Power (bhp)', value=100.0)
    mileage = st.sidebar.number_input('Mileage (kmpl)', value=18.0)
    engine = st.sidebar.number_input('Engine (CC)', value=1200)

    # Categorical Feature
    fuel = st.sidebar.selectbox('Fuel', car['fuel'].unique())
    seller_type = st.sidebar.selectbox('Seller Type', car['seller_type'].unique())
    transmission = st.sidebar.selectbox('Transmission', car['transmission'].unique())
    owner = st.sidebar.selectbox('Owner Type', car['owner'].unique())

    # Save to Dictionary
    data = {
        'year': year,
        'km_driven': km_driven,
        'seats': seats,
        'max_power (in bph)': max_power,
        'Mileage': mileage,
        'Engine (CC)': engine,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. Preprocessing (Encoding & Scaling) ---
def preprocess_input(input_df, original_df):
    combined_df = pd.concat([input_df, original_df.drop(columns=['selling_price'])], axis=0)
    
    # Label Encoding for categorical columns
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    for col in cat_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
    
    # Standardization (Scaling)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    num_cols = ['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']
    combined_df[num_cols] = scaler.fit_transform(combined_df[num_cols])
    
    return combined_df.iloc[[0]]

# --- 4. Main Page UI ---
st.title('🚗 Used Car Price Prediction')
st.subheader('The specifications that you entered:')
st.write(input_df)

if st.button('💰 Current Price Prediction'):
    if best_rfr and best_svr:
        clean_car = car.drop(columns=[col for col in ['Unnamed: 0', 'Mileage Unit', 'selling_price'] if col in car.columns])
        processed_input = preprocess_input(input_df, clean_car)
        
        # Prediction
        pred_rfr = best_rfr.predict(processed_input)
        pred_svr = best_svr.predict(processed_input)
        
        # The target (selling_price) in the original dataset is scaled, so it need to inverse transform
        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler()
        target_scaler.fit(car[['selling_price']])
        
        real_price_rfr = target_scaler.inverse_transform(pred_rfr.reshape(-1, 1))[0][0]
        real_price_svr = target_scaler.inverse_transform(pred_svr.reshape(-1, 1))[0][0]
        
        # Tampilan Hasil
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest Prediction", f"$ {real_price_rfr:,.0f}")
        with col2:
            st.metric("SVR Prediction", f"$ {real_price_svr:,.0f}")
            
        avg_price = (real_price_rfr + real_price_svr) / 2
        st.success(f"### Average Price Estimate: $ {avg_price:,.0f}")
    else:
        st.error(".joblib model not found. Make sure the model file is in the same directory.")

# Show reference data if checked
if st.checkbox('See Reference Dataset'):
    st.write(car.head())
