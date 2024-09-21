import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title of the app
st.title('Student Performance Prediction using Random Forest')

# Allow the user to upload the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Show the dataset to the user
    st.write("Dataset Preview:")
    st.write(data.head())

    # Ensure the dataset contains only numeric data
    if not data.select_dtypes(include='number').shape[1]:
        st.error("The dataset does not contain any numeric columns. Please upload a dataset with numeric features.")
    else:
        # Handle missing values by filling with the median (or you can choose another strategy)
        data = data.fillna(data.median())

        # Select the target and features
        target = 'Performance Index'  # Assuming this is the target column based on your description
        features = [col for col in data.columns if col != target]

        # Ensure target is present in the dataset
        if target not in data.columns:
            st.error(f"'{target}' column not found in the dataset. Please check the dataset.")
        else:
            # Split the data into features (X) and target (y)
            X = data[features]
            y = data[target]

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Sidebar for model training options
            st.sidebar.subheader('Model Training')

            if 'model' not in st.session_state:
                st.session_state.model = None

            # Train the model only when the button is clicked
            if st.sidebar.button('Train Random Forest Model'):
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.write('### Model trained successfully!')

                # Model Evaluation
                y_pred = model.predict(X_test)
                st.write(f"*Mean Absolute Error:* {mean_absolute_error(y_test, y_pred):.2f}")
                st.write(f"*Mean Squared Error:* {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"*R-squared Score:* {r2_score(y_test, y_pred):.2f}")

            # Section for user input and prediction
            st.subheader('Make Predictions')

            # Allow user to input values for the selected features
            user_input = []
            for feature in features:
                user_input.append(st.number_input(f"Enter value for {feature}", step=1.0, format="%.2f"))

            # Predict based on user input if model is trained
            if st.button('Predict'):
                if st.session_state.model is not None:
                    prediction = st.session_state.model.predict([user_input])
                    st.write(f'### Predicted {target}: {prediction[0]:.2f}')
                else:
                    st.write('Please train the model before making predictions.')
else:
    st.write("Please upload a dataset to proceed.")
