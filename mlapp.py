import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit app layout
st.set_page_config(page_title="Student Performance App", layout="wide")

# Sidebar for user inputs
st.sidebar.title("Upload and Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader('C:\Users\hp\Downloads\RegressionML\Student_Performance.csv', type=["csv"])

# If file is uploaded, proceed
if uploaded_file is not None:
    # Read the CSV file into a pandas dataframe
    data = pd.read_csv(uploaded_file)
    
    # Display a checkbox to show the dataset
    if st.sidebar.checkbox("Show Dataset Preview"):
        st.write("### Dataset Preview", data.head())

    # Main area layout
    st.title("Student Performance Analysis")

    # Filter numeric columns for ridge plot
    numeric_columns = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Performance Index']

    # Sidebar selection for ridge plot features
    selected_columns = st.sidebar.multiselect("Select features for ridge plot", numeric_columns, default=numeric_columns)

    # If the user selects columns, proceed to visualization
    if selected_columns:
        data_numeric = data[selected_columns]

        # Convert to numeric and handle errors
        data_numeric_clean = data_numeric.apply(pd.to_numeric, errors='coerce')

        # Melt the cleaned data for ridge plot
        melted_data_clean = data_numeric_clean.melt(var_name='Feature', value_name='Value')

        # Show ridge plot in main area
        st.subheader("Ridge Plot of Selected Features")
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        g = sns.FacetGrid(melted_data_clean, row="Feature", hue="Feature", aspect=4, height=1.5, palette="Set1")
        
        # Add density plots
        g.map(sns.kdeplot, "Value", fill=True)
        
        # Customize plot appearance
        g.set_titles("{row_name}")
        g.set(yticks=[])
        g.despine(left=True)

        # Display the plot
        st.pyplot(plt)

    # Add other features like filtering by extracurricular activities
    if st.sidebar.checkbox("Filter by Extracurricular Activities"):
        st.write("### Filtered by Extracurricular Activities")
        extracurricular_filter = st.sidebar.selectbox("Select filter", data['Extracurricular Activities'].unique())
        filtered_data = data[data['Extracurricular Activities'] == extracurricular_filter]
        st.write(filtered_data)

else:
    # Message when no file is uploaded
    st.write("Please upload a CSV file to proceed.")
