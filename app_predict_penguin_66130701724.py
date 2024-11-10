
import streamlit as st
import pickle
import numpy as np

# Load the trained KNN model and encoders
with open('knn_penguin.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# App title and description
st.title("Penguin Species Prediction App")
st.write("Enter the penguin's physical characteristics below to predict its species.")

# Input fields for features
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=30.0, max_value=60.0, step=0.1)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=10.0, max_value=25.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=150.0, max_value=250.0, step=1.0)
body_mass_g = st.number_input("Body Mass (g)", min_value=2500.0, max_value=6500.0, step=50.0)

# Predict button
if st.button("Predict Species"):
    # Check if all input fields have values
    if all(value > 0 for value in [culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g]):
        # Prepare input data for prediction
        input_data = np.array([[culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g]])
        
        # Make prediction using the model
        prediction = model.predict(input_data)
        
        # Map the predicted species (from the model output) to species names using the encoder
        species = species_encoder.inverse_transform(prediction)[0]
        
        # Output prediction for species
        st.write(f"The predicted species is: **{species}**")
    else:
        st.write("Please enter valid values for all fields.")
