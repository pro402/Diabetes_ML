import streamlit as st
import numpy as np
import pickle

# Set the title of the app
st.title("Diabetes Prediction App")

# Load the model from disk
with open('Diabetes_lr.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Create input fields for each feature
pregnancies = st.number_input('Enter number of Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.number_input('Enter GLUCOSE', min_value=0, max_value=200, step=1)
blood_pressure = st.number_input('Enter BloodPressure', min_value=0, max_value=200, step=1)
skin_thickness = st.number_input('Enter SkinThickness', min_value=0, max_value=100, step=1)
insulin = st.number_input('Enter Insulin', min_value=0, max_value=800, step=1)
bmi = st.number_input('Enter BMI', min_value=0.0, max_value=70.0, step=0.1)
diabetes_pedigree_function = st.number_input('Enter DiabetesPedigreeFunction', min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input('Enter Age', min_value=0, max_value=120, step=1)

# Create a button to trigger prediction
if st.button('Predict'):
    # Create a list from the inputs
    lst = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    # Reshape the list to a 2D array
    new_row = np.array(lst).reshape(1,-1)
    # Make a prediction
    new_prediction = loaded_model.predict(new_row)
    # Print the prediction
    st.write(f'Prediction for new row using loaded model: {new_prediction}')