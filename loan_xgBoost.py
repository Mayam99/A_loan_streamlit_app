# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# Set page config
st.set_page_config(page_title="Loan Default Predictor", layout="wide")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('models/xgb_loan_default_model.pkl')

@st.cache_data
def load_data():
    # This is just for demo - in practice you might not load the full dataset
    return pd.read_csv('application_train.csv')

# Preprocessing function
def preprocess_data(df):
    # Drop columns with >50% missing values
    df = df.loc[:, df.isnull().mean() < 0.5]
    
    # Impute numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# Main app function
def main():
    st.title("Loan Default Prediction App")
    st.write("""
    This app predicts the likelihood of loan default using XGBoost machine learning model.
    """)
    
    # Load model
    model = load_model()
    
    # Sidebar for user input
    st.sidebar.header("User Input Features")
    
    # Load data for reference (just for demo - in production you wouldn't do this)
    try:
        df = load_data()
        df = preprocess_data(df)
        
        # Get feature names (excluding target and ID)
        feature_names = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
        
        # Create input fields for each feature
        input_data = {}
        for feature in feature_names[:20]:  # Just show first 20 for demo
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].median())
            
            input_data[feature] = st.sidebar.slider(
                f"{feature}",
                min_val,
                max_val,
                default_val
            )
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale features
        scaler = StandardScaler()
        df_features = df[feature_names]
        scaler.fit(df_features)
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        if st.sidebar.button('Predict'):
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            st.subheader('Prediction')
            if prediction[0] == 1:
                st.error('High Risk of Default')
            else:
                st.success('Low Risk of Default')
            
            st.subheader('Prediction Probability')
            st.write(f"Probability of default: {prediction_proba[0][1]:.2%}")
            
            # Show feature importance
            st.subheader('Top Influential Features')
            fig, ax = plt.subplots(figsize=(10, 6))
            xgb_plot_importance(model, max_num_features=10, importance_type='gain', ax=ax)
            st.pyplot(fig)
    
    except Exception as e:
        st.warning("Couldn't load dataset - running in demo mode with sample prediction")
        if st.sidebar.button('Predict (Demo)'):
            st.subheader('Sample Prediction')
            st.success('Low Risk of Default (Demo)')
            st.write("Probability of default: 12.34% (Demo)")

    # Model information section
    st.sidebar.header("Model Information")
    st.sidebar.write("""
    - **Algorithm**: XGBoost Classifier
    - **Training Data**: Home Credit dataset
    - **Target**: Loan default prediction (0 = no default, 1 = default)
    """)

if __name__ == '__main__':
    main()