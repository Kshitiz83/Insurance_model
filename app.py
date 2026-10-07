import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("🏥 Medical Insurance Premium Predictor")
st.write("Adjust the client metrics below to calculate the estimated annual premium cost.")

# 2. LOAD YOUR SAVED ML ARTIFACTS
# (st.cache_resource stops the app from reloading the files on every single click)
@st.cache_resource
def load_resources():
    model=joblib.load('lasso_insurance_model.pkl')
    scaler=joblib.load('insurance_scaler.pkl')
    return model,scaler

lasso_model,scaler=load_resources()
# 3. BUILD INTERACTIVE USER WIDGETS
# These capture inputs from the web browser screen
age=st.slider('client_age',min_value=18,max_value=100,value=30)
bmi=st.number_input('Body Mass Index (BMI)',min_value=10.0,max_value=60.0,value=25.0,step=0.1)
children=st.selectbox('Number of Dependent Children',options=[0,1,2,3,4,5])

# For categorical features, use radio buttons or selectboxes
is_smoker = st.radio("Does the client smoke?", options=["No", "Yes"])
gender = st.radio("Gender", options=["Female", "Male"])
region = st.selectbox("Residential Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])

# 4. THE LOGICAL BRAIN CHALLENGE: THE ACTION BUTTON
if st.button('Calculate Premium Cost'):
    # --- STEP A: MAP INTERACTION VALUES TO DUMMY VARIABLES ---
    # We must match the EXACT column structure that your X_train matrix had!
    smoker_yes= 1 if is_smoker=="Yes" else 0
    sex_male= 1 if gender=="Male" else 0

    # Map regions to match the 'drop_first=True' headers your dummies made
    region_northwest= 1 if region=="Northwest" else 0
    region_southeast= 1 if region=="Southeast" else 0
    region_southwest= 1 if region=="Southwest" else 0

    # --- STEP B: ENGINEER THE MANUALLY CREATED INTERACTION FEATURE ---
    high_bmi_smoker= 1 if (smoker_yes==1 and bmi>30) else 0

    # --- STEP C: PROCESS CONTINUOUS COLUMNS THROUGH THE SCALER ---
    # Remember our double bracket matrix viewport rule! The scaler expects a 2D layout.
    scaled_features=scaler.transform([[age,bmi,children]])
    scaled_age=scaled_features[0][0]
    scaled_bmi=scaled_features[0][1]
    scaled_children=scaled_features[0][2]

    # --- STEP D: ASSEMBLE THE EXACT FEATURE VECTOR ---
    # Match the order of columns in X: [age, bmi, children, smoker_yes, region_northwest...]
    # Hint: Double check the exact column ordering printed by your VIF block or X.columns!
    input_vector=np.array([[
        smoker_yes,  
        high_bmi_smoker,  
        sex_male,  
        region_southeast,  
        scaled_bmi,  
        region_northwest,  
        region_southwest,  
        scaled_age,  
        scaled_children 
    ]])

    # --- STEP E: PREDICT AND INVERSE TRANSFORM OUT OF LOG SPACE ---
    predicted_log=lasso_model.predict(input_vector)
    final_premium=np.expm1(predicted_log)[0]

    # 5. DISPLAY THE RESULT BEAUTIFULLY
    st.success(f"🔮 Estimated Annual Premium: ${final_premium:,.2f}")