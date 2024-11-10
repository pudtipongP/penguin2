import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and encoders
with open('knn_penguin.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# App title and description
st.title("Penguin Species Prediction App")
st.write("Enter the penguin's physical characteristics below to predict its species.")

# Input fields for features
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=30.0, max_value=60.0, step=0.1)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=10.0, max_value=25.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=150.0, max_value=250.0, step=1.0)
body_mass_g = st.number_input("Body Mass (g)", min_value=2500.0, max_value=6500.0, step=50.0)
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])  # Ensure these are consistent with the training data labels

# Predict button
if st.button("Predict Species"):
    # Prepare the input data
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })

    # Ensure that the sex column is in the correct format
    x_new['sex'] = x_new['sex'].str.upper()  # Convert 'Male' to 'MALE' and 'Female' to 'FEMALE'

    # Encode the categorical columns
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])

    # Ensure the features are in the correct order, matching what the model expects
    input_data = x_new[['island' , 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

    # Make the prediction
    prediction = model.predict(input_data)

    # Map the prediction back to the species name using the species encoder
    predicted_species = species_encoder.inverse_transform(prediction)

    # Output the predicted species
    st.write(f"The predicted species is: **{predicted_species[0]}**")
