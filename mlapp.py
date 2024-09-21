import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Define the URL to the CSV file on GitHub or use local path
csv_url = 'https://raw.githubusercontent.com/PolakiVenkataSaiVineeth/machinelearningapp/main/Student_Performance.csv'  # Path to uploaded file

try:
    # Load the dataset
    data = pd.read_csv(csv_url)

    # Display the dataset to the user
    st.write("Dataset Preview:")
    st.write(data.head())

    # Specify that 'Performance Index' is the target column
    target_column = 'Performance Index'  # Make sure this matches the column name exactly

    # Check if 'Performance Index' is in the dataset
    if target_column not in data.columns:
        st.error(f"The dataset does not contain a column named '{target_column}'. Please check the column names.")
    else:
        # Encode the 'Extracurricular Activities' column, which is categorical
        if 'Extracurricular Activities' in data.columns:
            label_encoder = LabelEncoder()
            data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])

        # Separate features (X) and target (y)
        X = data.drop(columns=[target_column])  # Features (all columns except 'Performance Index')
        y = data[target_column]  # Target ('Performance Index')

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
        st.write("Enter values to predict the 'Performance Index':")

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
            st.write(f"The predicted 'Performance Index' is: {prediction[0]:.2f}")
except Exception as e:
    st.error(f"The file '{csv_url}' was not found or an error occurred. Error: {e}")
