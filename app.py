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

# ============================================================================
# PROFESSIONAL STYLING - PREMIUM DESIGN
# ============================================================================

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* MAIN BACKGROUND */
    .main {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 50%, #f0f7ff 100%);
        min-height: 100vh;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-right: 2px solid #e0e8f5;
    }
    
    /* HERO SECTION */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.08);
        border: 2px solid #e0e8f5;
        margin-bottom: 2rem;
        animation: slideDown 0.6s ease-out;
    }
    
    .hero h1 {
        background: linear-gradient(135deg, #0066ff 0%, #00a3ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero p {
        font-size: 1.1em;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.8rem;
        line-height: 1.6;
    }
    
    /* METRIC CARDS */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.06);
        border: 1.5px solid #e0e8f5;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(59, 130, 246, 0.12);
        border-color: #0066ff;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2em;
        font-weight: 800;
        background: linear-gradient(135deg, #0066ff 0%, #00a3ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* INFO CARDS */
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        padding: 2rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.06);
        border: 1.5px solid #e0e8f5;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.7s ease-out;
    }
    
    .card:hover {
        box-shadow: 0 12px 36px rgba(59, 130, 246, 0.1);
        border-color: #0066ff;
    }
    
    /* HEADERS */
    h2 {
        color: #0a1428;
        font-size: 1.6em;
        font-weight: 700;
        margin-top: 1.5rem !important;
        margin-bottom: 1.2rem !important;
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: #1e3a8a;
        font-size: 1.3em;
        font-weight: 700;
        margin-top: 1rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* FORM ELEMENTS */
    .stSlider > div > div > div > input {
        background: linear-gradient(135deg, #e0f2ff 0%, #e6f2ff 100%);
        border: 2px solid #0066ff !important;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 2px solid #e0e8f5 !important;
        border-radius: 12px;
        color: #0a1428;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #0066ff !important;
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.1);
    }
    
    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #0066ff 0%, #00a3ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem !important;
        font-weight: 700;
        font-size: 1.05em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 20px rgba(0, 102, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0, 102, 255, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* RESULT BOXES */
    .result-box {
        padding: 1.8rem;
        border-radius: 16px;
        color: white;
        font-weight: 700;
        text-align: center;
        animation: resultPulse 0.6s ease-out;
    }
    
    .result-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.3);
        border: 2px solid #6ee7b7;
    }
    
    .result-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 12px 32px rgba(245, 158, 11, 0.3);
        border: 2px solid #fbbf24;
    }
    
    .result-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 12px 32px rgba(239, 68, 68, 0.3);
        border: 2px solid #fca5a5;
    }
    
    /* TEXT STYLES */
    .customer-info {
        font-size: 1.05em;
        color: #1e293b;
        margin: 0.8rem 0;
        font-weight: 500;
        padding: 0.8rem;
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 100%);
        border-left: 4px solid #0066ff;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .customer-info:hover {
        background: linear-gradient(135deg, #e6f2ff 0%, #d5e5ff 100%);
        border-left-color: #00a3ff;
    }
    
    /* ANIMATIONS */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes resultPulse {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* ICONS STYLING */
    .icon-large {
        font-size: 2.5em;
        margin-bottom: 0.5rem;
    }
    
    /* SIDEBAR HEADER */
    .sidebar-header {
        background: linear-gradient(135deg, #0066ff 0%, #00a3ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.4em;
        font-weight: 800;
        margin-bottom: 1.5rem;
    }
    
    /* DIVIDER */
    hr {
        border: none;
        border-top: 2px solid #e0e8f5;
        margin: 2rem 0;
    }
    
    /* CAPTION TEXT */
    .stCaption {
        color: #64748b !important;
        font-size: 0.9em !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model.pkl"

if not model_path.exists():
    st.error(f"❌ Model file not found: {model_path.name}")
    st.info("Make sure 'model.pkl' is in the same directory as this script")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ============================================================================
# HERO SECTION
# ============================================================================

st.markdown("""
<div class="hero">
    <h1>✈️ Travel Customer Churn Predictor</h1>
    <p>Advanced machine learning model to predict customer churn risk in the travel industry. Make data-driven retention decisions instantly.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# KEY METRICS
# ============================================================================

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📊 Dataset Size</div>
        <div class="metric-value">954</div>
        <div style="font-size: 0.85em; color: #64748b; margin-top: 0.5rem;">Training samples</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">⚠️ Baseline Churn</div>
        <div class="metric-value">23.5%</div>
        <div style="font-size: 0.85em; color: #64748b; margin-top: 0.5rem;">Historical rate</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🤖 Model Type</div>
        <div class="metric-value">RF</div>
        <div style="font-size: 0.85em; color: #64748b; margin-top: 0.5rem;">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR - CUSTOMER INPUTS
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">📋 Customer Profile</div>
    """, unsafe_allow_html=True)
    
    st.markdown("Enter customer details below:")
    
    age = st.slider(
        "🎂 Age",
        min_value=18,
        max_value=75,
        value=31,
        step=1,
        help="Customer's age in years"
    )
    
    frequent = st.selectbox(
        "✈️ Frequent Flyer",
        ["No", "Yes"],
        help="Is this customer a frequent flyer?"
    )
    
    income = st.selectbox(
        "💰 Income Class",
        ["Low Income", "Middle Income"],
        help="Customer's annual income classification"
    )
    
    services = st.slider(
        "🎁 Services Opted",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help="Number of services the customer has opted for"
    )
    
    social = st.selectbox(
        "📱 Social Media Sync",
        ["No", "Yes"],
        help="Is their account synced with social media?"
    )
    
    hotel = st.selectbox(
        "🏨 Booked Hotel",
        ["No", "Yes"],
        help="Has the customer booked a hotel?"
    )
    
    st.caption("✨ All fields are required. Click 'Predict Churn' to get results.")

# ============================================================================
# MAIN CONTENT - CUSTOMER PROFILE & PREDICTION
# ============================================================================

st.markdown("### Customer Information & Prediction")

left_col, right_col = st.columns([1.3, 1], gap="medium")

# ============================================================================
# LEFT COLUMN - CUSTOMER PROFILE
# ============================================================================

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="margin-bottom: 1.2rem;">👤 Customer Profile Summary</h3>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="customer-info">
        <strong>Age:</strong> {age} years old
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="customer-info">
        <strong>Travel Status:</strong> {'✈️ Frequent Flyer' if frequent == 'Yes' else '🚫 Occasional Traveler'}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="customer-info">
        <strong>Income Class:</strong> {income}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="customer-info">
        <strong>Services Opted:</strong> {services} out of 6 services
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="customer-info">
        <strong>Social Media Sync:</strong> {'✅ Connected' if social == 'Yes' else '❌ Not Connected'}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="customer-info">
        <strong>Hotel Booking:</strong> {'✅ Has booked' if hotel == 'Yes' else '❌ No booking'}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RIGHT COLUMN - PREDICTION SECTION
# ============================================================================

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="margin-bottom: 1.5rem;">🎯 Churn Prediction</h3>
    """, unsafe_allow_html=True)
    
    # PREDICT BUTTON
    predict_clicked = st.button("🔮 Predict Churn Risk", use_container_width=True, key="predict_btn")
    
    if predict_clicked:
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'FrequentFlyer': [frequent],
            'AnnualIncomeClass': [income],
            'ServicesOpted': [services],
            'AccountSyncedToSocialMedia': [social],
            'BookedHotelOrNot': [hotel]
        })
        
        # Encode categorical variables
        le = LabelEncoder()
        cat_cols = ['FrequentFlyer', 'AnnualIncomeClass', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']
        
        for col in cat_cols:
            input_data[col] = le.fit_transform(input_data[col])
        
        # Get prediction
        try:
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            # Display result
            st.markdown("<br>", unsafe_allow_html=True)
            
            if pred == 1:  # HIGH CHURN RISK
                st.markdown(f"""
                <div class="result-box result-error">
                    <div style="font-size: 2.5em; margin-bottom: 0.5rem;">🚨</div>
                    <div style="font-size: 1.3em;">HIGH CHURN RISK</div>
                    <div style="font-size: 1.8em; margin-top: 0.5rem;">{prob:.0%}</div>
                    <div style="font-size: 0.9em; margin-top: 1rem; opacity: 0.9;">Probability of churn</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("⚠️ **Action Required:** This customer shows high churn indicators. Consider immediate retention strategies.")
                
            else:  # LOW CHURN RISK
                st.markdown(f"""
                <div class="result-box result-success">
                    <div style="font-size: 2.5em; margin-bottom: 0.5rem;">✅</div>
                    <div style="font-size: 1.3em;">LOW CHURN RISK</div>
                    <div style="font-size: 1.8em; margin-top: 0.5rem;">{(1-prob):.0%}</div>
                    <div style="font-size: 0.9em; margin-top: 1rem; opacity: 0.9;">Likelihood to stay</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✨ **Great News:** This customer is likely to remain engaged and satisfied.")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0; font-size: 0.95em;">
    <p>🔒 <strong>Enterprise-Grade Security</strong> • All data processed locally</p>
    <p style="margin-top: 0.5rem;">Powered by Random Forest ML • Built with Streamlit</p>
    <p style="margin-top: 1rem; font-size: 0.85em; opacity: 0.7;">© 2024 Travel Churn Predictor • Production Ready</p>
</div>
""", unsafe_allow_html=True)
 fig(
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
