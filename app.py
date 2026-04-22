import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

st.set_page_config(
    page_title="Travel Churn Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CSS =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 50%, #f0f7ff 100%);
}

/* FIX TEXT VISIBILITY */
label, .stMarkdown, .stText, .stCaption {
    color: #0a1428 !important;
    font-weight: 500;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #0a1428 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid #e0e8f5;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e0e8f5;
}

/* Button */
.stButton > button {
    background: #0066ff;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}

/* Result */
.result-box {
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
}

.success { background: #10b981; }
.error { background: #ef4444; }

</style>
""", unsafe_allow_html=True)

# ========================= MODEL =========================
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model.pkl"

if not model_path.exists():
    st.error("Model file not found")
    st.stop()

model = joblib.load(model_path)

# ========================= HEADER =========================
st.title("Travel Customer Churn Predictor")
st.write("Predict customer churn using machine learning")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown("<h3>Customer Profile</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#0a1428;'>Enter customer details below:</p>", unsafe_allow_html=True)

    age = st.slider("Age", 18, 75, 30)

    frequent = st.selectbox("Frequent Flyer", ["No", "Yes"])

    income = st.selectbox("Income Class", ["Low Income", "Middle Income"])

    services = st.slider("Services Opted", 1, 6, 2)

    social = st.selectbox("Social Media Sync", ["No", "Yes"])

    hotel = st.selectbox("Booked Hotel", ["No", "Yes"])

# ========================= MAIN =========================
col1, col2 = st.columns(2)

# LEFT - PROFILE
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Customer Summary")

    st.write(f"Age: {age}")
    st.write(f"Frequent Flyer: {frequent}")
    st.write(f"Income Class: {income}")
    st.write(f"Services Opted: {services}")
    st.write(f"Social Media: {social}")
    st.write(f"Hotel Booking: {hotel}")

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT - PREDICTION
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction")

    if st.button("Predict Churn"):

        input_data = pd.DataFrame({
            'Age': [age],
            'FrequentFlyer': [frequent],
            'AnnualIncomeClass': [income],
            'ServicesOpted': [services],
            'AccountSyncedToSocialMedia': [social],
            'BookedHotelOrNot': [hotel]
        })

        le = LabelEncoder()
        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                input_data[col] = le.fit_transform(input_data[col])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.markdown(f"<div class='result-box error'><h3>HIGH CHURN RISK</h3><h2>{prob:.0%}</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box success'><h3>LOW CHURN RISK</h3><h2>{(1-prob):.0%}</h2></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ========================= FOOTER =========================
st.markdown("---")
st.write("Built with Streamlit | Machine Learning Model")
