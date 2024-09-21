import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Define the path to the CSV file on your local machine
csv_path = 'C:/Users/hp/Downloads/RegressionML' # Local file path

# Check if the file exists before attempting to read it
if os.path.exists(csv_path):
    # Load the dataset
    data = pd.read_csv(csv_path)

    # Display the dataset to the user
    st.write("Dataset Preview:")
    st.write(data.head())

    # Specify that 'performance index' is the target column
    target_column = 'performance index'  # Make sure this matches the column name exactly

    # Check if 'performance index' is in the dataset
    if target_column not in data.columns:
        st.error(f"The dataset does not contain a column named '{target_column}'. Please check the column names.")
    else:
        # Separate features (X) and target (y)
        X = data.drop(columns=[target_column])  # Features (all columns except 'performance index')
        y = data[target_column]  # Target ('performance index')

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple model (Linear Regression)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Test the model and display performance
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Model Mean Squared Error: {mse:.2f}")

        # Create input fields for the user to enter new data for prediction
        st.write("Enter values to predict the 'performance index':")

        # Dynamically generate input fields based on the features in the dataset
        user_input = {}
        for feature in X.columns:
            # For numeric values, use st.number_input; default value is set to 0.0
            user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

        # Convert user inputs into a DataFrame for prediction
        input_df = pd.DataFrame([user_input])

        # Make predictions based on user inputs
        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.write(f"The predicted 'performance index' is: {prediction[0]:.2f}")
else:
    st.error(f"The file '{csv_path}' was not found. Please check the file path.")

