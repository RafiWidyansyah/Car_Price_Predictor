import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Column Variable Definition ---
categoric = ['fuel', 'seller_type', 'transmission', 'owner'] 
numeric = ['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']
target = 'selling_price'

# --- 2. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('processes2.csv')
    return df

car = load_data()

# --- 3. Preprocessing Functions ---
def preprocess_data(df):
    # Delete unnecessary columns
    cols_to_drop = [col for col in ['Unnamed: 0', 'Mileage Unit'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Scaler and Encoder Initialization (For inverse transform needs later)
    scaler = StandardScaler()
    le = LabelEncoder()
    
    # Copy dataframe to be processed
    processed_df = df.copy()
    
    # Encoding Categorical
    for cat in [c for c in categoric if c in processed_df.columns]:
        processed_df[cat] = le.fit_transform(processed_df[cat])
        
    # Standardize Numerical (Including the target to facilitate inverse in this example)
    num_cols = [n for n in numeric + [target] if n in processed_df.columns]
    processed_df[num_cols] = scaler.fit_transform(processed_df[num_cols])
    
    return processed_df, scaler

processed_car, fitted_scaler = preprocess_data(car)

# --- 4. Load Models ---
@st.cache_resource
def load_models():
    try:
        rfr = joblib.load('best_random_forest_regressor_model.joblib')
        svr = joblib.load('best_SVR_Model.joblib')
        return rfr, svr
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

best_rfr, best_svr = load_models()

# --- 5. Prediction Logic ---
def predict_price(df, scaler):
    # Pisahkan Fitur dan Target
    X = df.drop(target, axis=1)
    y = df[target]

    # Split data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make a Prediction Using a Loaded Model
    y_pred_scaled_rfr = best_rfr.predict(X_test)
    y_pred_scaled_svr = best_svr.predict(X_test)

    sp_mean = scaler.mean_[-1]
    sp_std = scaler.scale_[-1]

    # Inverse Transform to the original price
    y_pred_original_rfr = (y_pred_scaled_rfr * sp_std) + sp_mean
    y_pred_original_svr = (y_pred_scaled_svr * sp_std) + sp_mean

    return y_pred_original_rfr, y_pred_original_svr

# --- 6. Streamlit UI ---
st.title('🚗 Car Price Prediction App')

# Sidebar for Feature Filter
with st.sidebar:
    st.header("Data Configuration")
    feature_names = processed_car.columns.tolist()
    selected_features = st.multiselect(
        'Choose a Feature for Prediction : ',
        options=feature_names,
        default=feature_names
    )

# Filter DataFrame based on user's choice
if target not in selected_features:
    st.warning(f"Make sure '{target}' is still selected to carry out this prediction test function.")

filtered_data = processed_car[selected_features]

# Tombol Prediksi
if st.button('Run Test Prediction'):
    if best_rfr and best_svr:
        y_rfr, y_svr = predict_price(filtered_data, fitted_scaler)

        st.subheader("Prediction Result :")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RF Mean Prediction", f"Rp {y_rfr.mean():,.2f}")
        with col2:
            st.metric("SVR Mean Prediction", f"Rp {y_svr.mean():,.2f}")

        avg_combined = (y_rfr + y_svr) / 2
        st.info(f"Combined Prediction Average: $ {avg_combined.mean():,.2f}")
        
        # Show the top 5 result table
        res_df = pd.DataFrame({'RF': y_rfr, 'SVR': y_svr}).head()
        st.table(res_df)
    else:
        st.error("Model is not available.")

# --- Display Raw Data ---
if st.checkbox('Show Raw Data'):
    st.subheader("Raw Car Data:")
    st.dataframe(car.head(10))
