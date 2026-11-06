import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# 1. Load the trained models and both encoders efficiently
@st.cache_resource
def load_assets():
    churn_model = tf.keras.models.load_model('model.h5')
    salary_model = tf.keras.models.load_model('regression_model.h5')
    
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    # Load BOTH scalers to prevent feature mismatch
    with open('scaler_churn.pkl', 'rb') as file:
        scaler_churn = pickle.load(file)
        
    with open('scaler_salary.pkl', 'rb') as file:
        scaler_salary = pickle.load(file)
        
    return churn_model, salary_model, label_encoder_gender, onehot_encoder_geo, scaler_churn, scaler_salary

churn_model, salary_model, label_encoder_gender, onehot_encoder_geo, scaler_churn, scaler_salary = load_assets()

st.title('🏦 Bank AI: Churn & Salary Predictor')

tab1, tab2 = st.tabs(["🔴 Churn Prediction", "💵 Salary Estimation"])

# ==========================================
# TAB 1: CHURN PREDICTION
# ==========================================
with tab1:
    st.header('Predict Customer Attrition')
    
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='c_geo')
    gender = st.selectbox('Gender', label_encoder_gender.classes_, key='c_gender')
    age = st.slider('Age', 18, 92, key='c_age')
    balance = st.number_input('Balance ($)', min_value=0.0, value=60000.0, step=1000.0, key='c_bal')
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600, step=1, key='c_credit')
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0, step=1000.0, key='c_sal')
    tenure = st.slider('Tenure (Years)', 0, 10, key='c_tenure')
    num_of_products = st.slider('Number of Products', 1, 4, key='c_prod')
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='c_card')
    is_active_member = st.selectbox('Is Active Member', [0, 1], key='c_active')

    if st.button("Predict Churn Probability"):
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Use the specific churn scaler
        input_data_scaled = scaler_churn.transform(input_data)
        prediction = churn_model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.write(f'**Churn Probability:** {prediction_proba:.2f}')

        if prediction_proba > 0.5:
            st.error('⚠️ High Risk: The customer is likely to churn.')
        else:
            st.success('✅ Low Risk: The customer is not likely to churn.')

# ==========================================
# TAB 2: SALARY ESTIMATION (REGRESSION)
# ==========================================
with tab2:
    st.header('Estimate Customer Salary')
    
    geography_s = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='s_geo')
    gender_s = st.selectbox('Gender', label_encoder_gender.classes_, key='s_gender')
    age_s = st.slider('Age', 18, 92, key='s_age')
    balance_s = st.number_input('Balance ($)', min_value=0.0, value=60000.0, step=1000.0, key='s_bal')
    credit_score_s = st.number_input('Credit Score', min_value=300, max_value=850, value=600, step=1, key='s_credit')
    exited_s = st.selectbox('Did Customer Exit (Churn)?', [0, 1], key='s_exited')
    tenure_s = st.slider('Tenure (Years)', 0, 10, key='s_tenure')
    num_of_products_s = st.slider('Number of Products', 1, 4, key='s_prod')
    has_cr_card_s = st.selectbox('Has Credit Card', [0, 1], key='s_card')
    is_active_member_s = st.selectbox('Is Active Member', [0, 1], key='s_active')

    if st.button("Estimate Salary"):
        input_data_s = pd.DataFrame({
            'CreditScore': [credit_score_s],
            'Gender': [label_encoder_gender.transform([gender_s])[0]],
            'Age': [age_s],
            'Tenure': [tenure_s],
            'Balance': [balance_s],
            'NumOfProducts': [num_of_products_s],
            'HasCrCard': [has_cr_card_s],
            'IsActiveMember': [is_active_member_s],
            'Exited': [exited_s] 
        })

        geo_encoded_s = onehot_encoder_geo.transform([[geography_s]]).toarray()
        geo_encoded_df_s = pd.DataFrame(geo_encoded_s, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data_s = pd.concat([input_data_s.reset_index(drop=True), geo_encoded_df_s], axis=1)

        # Use the specific salary scaler
        input_data_scaled_s = scaler_salary.transform(input_data_s)
        salary_prediction = salary_model.predict(input_data_scaled_s)
        estimated_salary_val = salary_prediction[0][0]
        
        st.divider()
        st.success(f'💵 **Estimated Annual Salary:** ${estimated_salary_val:,.2f}')