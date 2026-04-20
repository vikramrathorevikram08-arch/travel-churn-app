import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

st.set_page_config(
    page_title="Travel Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: white;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid #e8eef7;
        margin-bottom: 20px;
    }
    .card {
        background: white;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid #e8eef7;
    }
    .result-box {
        padding: 18px;
        border-radius: 16px;
        color: white;
        font-weight: 600;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model.pkl"

if not model_path.exists():
    st.error(f"Model file not found: {model_path.name}")
    st.stop()

model = joblib.load(model_path)

st.markdown("""
<div class="hero">
    <h1 style="margin-bottom: 0;">✈️ Travel Customer Churn Predictor</h1>
    <p style="font-size: 18px; color: #555; margin-top: 8px;">
        Predict whether a customer is likely to churn using a trained Random Forest model.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset Size", "954")
with col2:
    st.metric("Churn Rate", "23.5%")
with col3:
    st.metric("Model", "Random Forest")

st.markdown("### Enter Customer Details")

with st.sidebar:
    st.header("Customer Inputs")
    age = st.slider("Age", 27, 38, 31)
    frequent = st.selectbox("Frequent Flyer", ["No", "Yes"])
    income = st.selectbox("Income Class", ["Low Income", "Middle Income"])
    services = st.slider("Services Opted", 1, 6, 2)
    social = st.selectbox("Social Media Sync", ["No", "Yes"])
    hotel = st.selectbox("Booked Hotel", ["No", "Yes"])

    st.caption("Fill details and click predict to see churn chance.")

left, right = st.columns([1.2, 0.8])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Customer Profile")
    st.write(f"**Age:** {age}")
    st.write(f"**Frequent Flyer:** {frequent}")
    st.write(f"**Income Class:** {income}")
    st.write(f"**Services Opted:** {services}")
    st.write(f"**Social Media Synced:** {social}")
    st.write(f"**Booked Hotel:** {hotel}")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")
    if st.button("🔮 Predict Churn", use_container_width=True):
        input_data = pd.DataFrame({
            'Age': [age],
            'FrequentFlyer': [frequent],
            'AnnualIncomeClass': [income],
            'ServicesOpted': [services],
            'AccountSyncedToSocialMedia': [social],
            'BookedHotelOrNot': [hotel]
        })

        le = LabelEncoder()
        cat_cols = ['FrequentFlyer', 'AnnualIncomeClass', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']
        for col in cat_cols:
            input_data[col] = le.fit_transform(input_data[col])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f"⚠️ High Churn Risk: {prob:.0%}")
        else:
            st.success(f"✅ Likely to Stay: {100 - prob:.0%}")

    st.markdown('</div>', unsafe_allow_html=True)
