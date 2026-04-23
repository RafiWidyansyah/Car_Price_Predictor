import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    except EOFError:
        st.error("🚨 **Error:** One of your model files (.joblib) is corrupted or empty.")
        st.info("Try re-uploading your model files to your repository.")
        st.stop() # Stops the app from running further
    except FileNotFoundError:
        st.error("🚨 **Error:** Model files not found. Make sure they are in the same folder as app.py.")
        st.stop()

car = load_data()
best_rfr, best_svr = load_models()

# --- 2. Preprocessing Logic (Setup) ---
# We need the scaler and encoders to transform USER input later
scaler = StandardScaler()
scaler.fit(car[['selling_price']]) 

# Identify feature types
numeric_cols = ['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']
# Any column not in numeric, selling_price, or junk is categorical
categoric_cols = [col for col in car.columns if col not in numeric_cols + ['selling_price', 'Unnamed: 0', 'Mileage Unit']]

# --- 3. Prediction Function ---
def predict_price(input_df):
    # This function expects the dataframe to already be encoded/scaled if the model requires it.
    # However, for simplicity here, we assume the model handles the raw features 
    # or you apply transformation before calling this.
    y_pred_rfr = best_rfr.predict(input_df)
    y_pred_svr = best_svr.predict(input_df)

    # Inverse scaling for selling_price
    selling_price_mean = scaler.mean_[0]
    selling_price_std = np.sqrt(scaler.var_[0])

    res_rfr = (y_pred_rfr * selling_price_std) + selling_price_mean
    res_svr = (y_pred_svr * selling_price_std) + selling_price_mean

    return res_rfr, res_svr

# --- 4. Streamlit UI ---
st.title('🚗 Car Price Prediction App')

with st.sidebar:
    st.header("Input Car Features")
    
    user_data = {}
    
    # Generate dynamic inputs based on the columns
    for col in car.columns:
        if col in ['selling_price', 'Unnamed: 0', 'Mileage Unit']:
            continue
            
        if col in numeric_cols:
            # Numerical Input: Use min/max from data to set reasonable bounds
            min_val = float(car[col].min())
            max_val = float(car[col].max())
            default_val = float(car[col].mean())
            
            user_data[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=default_val)
        
        else:
            # Categorical Input: Create a dropdown from unique values
            options = car[col].unique().tolist()
            user_data[col] = st.selectbox(f"Select {col}", options=options)

# Convert user input into a DataFrame
input_df = pd.DataFrame([user_data])

# --- 5. Encoding and Scaling User Input ---
# This is CRITICAL: Models cannot read text. We must encode the input_df exactly like the training data.
processed_input = input_df.copy()

# Encoding categorical inputs
for col in categoric_cols:
    le = LabelEncoder()
    # Fit on original data to ensure the mapping is the same
    le.fit(car[col].astype(str))
    processed_input[col] = le.transform(processed_input[col].astype(str))

# Scaling numerical inputs (excluding the target)
# Note: If your model was trained on scaled features, you must scale these too.
# For now, we follow your script's logic.

# --- 6. Prediction Logic ---
if st.button('Predict Price'):
    try:
        # Ensure the columns are in the exact order the model expects
        # We use the columns from the training set (minus the target)
        model_features = [c for c in car.columns if c not in ['selling_price', 'Unnamed: 0', 'Mileage Unit']]
        processed_input = processed_input[model_features]

        res_rfr, res_svr = predict_price(processed_input)

        st.subheader("Prediction Results:")
        col1, col2 = st.columns(2)
        col1.metric("Random Forest", f"${res_rfr[0]:,.2f}")
        col2.metric("SVR", f"${res_svr[0]:,.2f}")
        
        avg = (res_rfr[0] + res_svr[0]) / 2
        st.success(f"**Estimated Market Value:** ${avg:,.2f}")
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check if your model expects more/different features than what is provided.")

if st.checkbox('Show Reference Data'):
    st.write("Using these unique values/ranges for inputs:")
    st.dataframe(car.describe())
