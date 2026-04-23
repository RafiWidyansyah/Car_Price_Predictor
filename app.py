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
        st.stop()
    except FileNotFoundError:
        st.error("🚨 **Error:** Model files not found. Make sure they are in the same folder as app.py.")
        st.stop()

car = load_data()
best_rfr, best_svr = load_models()

# --- 2. Preprocessing Logic (Setup) ---
scaler = StandardScaler()
scaler.fit(car[['selling_price']])

numeric_cols   = ['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']
integer_cols   = ['year', 'km_driven', 'seats', 'Engine (CC)']  # whole numbers only
float_cols     = ['max_power (in bph)', 'Mileage']               # decimals allowed
categoric_cols = [
    col for col in car.columns
    if col not in numeric_cols + ['selling_price', 'Unnamed: 0', 'Mileage Unit']
]

# --- 3. Prediction Function ---
def predict_price(input_df):
    y_pred_rfr = best_rfr.predict(input_df)
    y_pred_svr = best_svr.predict(input_df)

    selling_price_mean = scaler.mean_[0]
    selling_price_std  = np.sqrt(scaler.var_[0])

    res_rfr = (y_pred_rfr * selling_price_std) + selling_price_mean
    res_svr = (y_pred_svr * selling_price_std) + selling_price_mean

    return res_rfr, res_svr

# --- 4. Streamlit UI ---
st.title('🚗 Used Car Price Prediction App')

with st.sidebar:
    st.header("Input Car Features")

    user_data         = {}
    validation_errors = []

    for col in car.columns:
        if col in ['selling_price', 'Unnamed: 0', 'Mileage Unit']:
            continue

        if col in integer_cols:
            # ── INTEGER input ──────────────────────────────────────
            raw = st.text_input(
                label=col,
                value="",
                placeholder=f"e.g. {int(car[col].mean())}",
                help=f"Whole number only. Dataset range: {int(car[col].min())} – {int(car[col].max())}"
            )
            if raw == "":
                user_data[col] = None
                validation_errors.append(f"**{col}** is required.")
            else:
                try:
                    parsed = float(raw)
                    if not parsed.is_integer():
                        raise ValueError("decimal not allowed")
                    user_data[col] = int(parsed)
                except ValueError:
                    user_data[col] = None
                    validation_errors.append(
                        f"**{col}** must be a whole number with no decimals (got: `{raw}`)."
                    )

        elif col in float_cols:
            # ── FLOAT input ────────────────────────────────────────
            raw = st.text_input(
                label=col,
                value="",
                placeholder=f"e.g. {float(car[col].mean()):.1f}",
                help=f"Decimals allowed. Dataset range: {float(car[col].min()):.1f} – {float(car[col].max()):.1f}"
            )
            if raw == "":
                user_data[col] = None
                validation_errors.append(f"**{col}** is required.")
            else:
                try:
                    user_data[col] = float(raw)
                except ValueError:
                    user_data[col] = None
                    validation_errors.append(
                        f"**{col}** must be a numeric value (got: `{raw}`)."
                    )

        else:
            # ── DROPDOWN for categorical fields ────────────────────
            options = car[col].unique().tolist()
            user_data[col] = st.selectbox(f"Select {col}", options=options)

# --- 5. Validation feedback ---
if validation_errors:
    with st.expander("⚠️ Please fix the following inputs", expanded=True):
        for err in validation_errors:
            st.markdown(f"- {err}")

# --- 6. Build & encode input DataFrame ---
input_df        = pd.DataFrame([user_data])
processed_input = input_df.copy()

for col in categoric_cols:
    le = LabelEncoder()
    le.fit(car[col].astype(str))
    processed_input[col] = le.transform(processed_input[col].astype(str))

# --- 7. Prediction Button ---
predict_clicked = st.button('Predict Price', disabled=bool(validation_errors))

if predict_clicked:
    try:
        model_features  = [c for c in car.columns if c not in ['selling_price', 'Unnamed: 0', 'Mileage Unit']]
        processed_input = processed_input[model_features]

        # Ensure correct dtypes before prediction
        for col in integer_cols:
            processed_input[col] = processed_input[col].astype(int)
        for col in float_cols:
            processed_input[col] = processed_input[col].astype(float)

        res_rfr, res_svr = predict_price(processed_input)

        st.subheader("Prediction Results:")
        c1, c2 = st.columns(2)
        c1.metric("Random Forest", f"${res_rfr[0]:,.2f}")
        c2.metric("SVR",           f"${res_svr[0]:,.2f}")

        avg = (res_rfr[0] + res_svr[0]) / 2
        st.success(f"**Estimated Market Value:** ${avg:,.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Check if your model expects more/different features than what is provided.")

if st.checkbox('Show Reference Data'):
    st.write("Using these unique values/ranges for inputs:")
    st.dataframe(car.describe())
