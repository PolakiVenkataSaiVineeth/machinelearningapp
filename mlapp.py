import streamlit as st
import pandas as pd
import joblib

# Load the Random Forest model
model = joblib.load('random_forest_model.pkl')  # Make sure this path is correct

# Streamlit app layout
st.title("Random Forest Model Deployment")

# Collect user input for model predictions
st.header("Input Features")

# Modify these inputs according to your dataset features
feature1 = st.number_input('Feature 1', min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.number_input('Feature 2', min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.number_input('Feature 3', min_value=0.0, max_value=100.0, value=50.0)
feature4 = st.number_input('Feature 4', min_value=0.0, max_value=100.0, value=50.0)

# Create a dataframe from user input
input_data = pd.DataFrame({
    'Feature1': [feature1],
    'Feature2': [feature2],
    'Feature3': [feature3],
    'Feature4': [feature4]
})

# Predict using the Random Forest model
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
