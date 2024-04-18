import streamlit as st
import numpy as np
import pickle
import time

# Set the title of the app
st.title("Diabetes Prediction App")
st.write("This works on the Logistic Regression model for classifing")
st.write("The output may not be 100% accurate as the accuracy is only 0.8397435897435898")
# Load the model from disk
with open('Notebooks/Diabetes_lr.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Create input fields for each feature
pregnancies = st.slider('Enter number of Pregnancies',0,20,0)
glucose = st.slider('Enter GLUCOSE',0,200,)
blood_pressure = st.slider('Enter BloodPressure',0,400,70)
skin_thickness = st.slider('Enter SkinThickness',0,30,10)
insulin = st.slider('Enter Insulin',0,800,20)
bmi = st.slider('Enter BMI',0.0,60.0,20.0)
diabetes_pedigree_function = st.slider('Enter DiabetesPedigreeFunction', 0.00, 2.50, 0.0)
age = st.slider('How old are you?', 0, 130, 25)

# Create a button to trigger prediction
if st.button('Predict'):
    # Create a list from the inputs
    lst = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    # Reshape the list to a 2D array
    new_row = np.array(lst).reshape(1,-1)
    # Make a prediction
    new_prediction = loaded_model.predict(new_row)
    # Print the prediction
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')
    st.balloons()
    st.write(f'Prediction for the data inserted : {new_prediction}')
    if new_prediction == 0:
      st.write("It seems like you are out of danger for now ...")
    else:
      st.write("You might have Diabetes do go for proper health checkup ...")
