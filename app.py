import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load('model.pkl')
st.title("✈️ Travel Churn Predictor")

age = st.slider("Age", 27, 38, 31)
frequent = st.selectbox("Frequent Flyer", ["No", "Yes"])
income = st.selectbox("Income", ["Low Income", "Middle Income"])
services = st.slider("Services", 1, 6, 2)
social = st.selectbox("Social Media", ["No", "Yes"])
hotel = st.selectbox("Hotel", ["No", "Yes"])

if st.button("🔮 Predict"):
    input_data = pd.DataFrame({
        'Age': [age], 'FrequentFlyer': [frequent],
        'AnnualIncomeClass': [income], 'ServicesOpted': [services],
        'AccountSyncedToSocialMedia': [social], 'BookedHotelOrNot': [hotel]
    })
    
    le = LabelEncoder()
    cat_cols = ['FrequentFlyer','AnnualIncomeClass','AccountSyncedToSocialMedia','BookedHotelOrNot']
    for col in cat_cols:
        input_data[col] = le.fit_transform(input_data[col])
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Result: {'🛑 CHURN' if pred==1 else '✅ STAY'} ({prob:.0%})")
