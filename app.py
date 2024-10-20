import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

min_max_scaler = joblib.load('min_max_scaler.pkl')

# Streamlit session state to keep track of gender selection across pages
if 'gender' not in st.session_state:
    st.session_state.gender = None

# Helper function to add style to sections
def styled_header(title, subtitle=None):
    st.markdown(f"<h1 style='color: #4CAF50;'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<h3 style='color: #555;'>{subtitle}</h3>", unsafe_allow_html=True)

# Page 1: Gender Selection
if st.session_state.gender is None or st.session_state.gender == "Select your gender":
    styled_header("Diabetes Prediction App")
    
    # Add explanation text with formatting
    # st.markdown("### Help us provide accurate predictions by selecting the appropriate gender category.")
    st.session_state.gender = st.selectbox("Select your gender", options=["Select your gender", "Male", "Female"])
    
    # Check if gender is selected, then rerun
    if st.session_state.gender != "Select your gender":
        st.experimental_rerun()

# Page 2: Gender-Specific Questions
else:
    styled_header(f"Questionnaire for {st.session_state.gender} Patients")
    st.markdown("Please fill out the details carefully. Accurate information helps in better prediction.")

    # Option to go back to gender selection
    if st.button("Back to Gender Selection", key="back"):
        st.session_state.gender = None
        st.experimental_rerun()

    # Initialize the dictionary to store gender-specific data
    gender_specific_data = {}

    # Helper function for number input with better appearance
    def number_input_with_none(label, min_value=None):
        user_input = st.text_input(label)
        return float(user_input) if user_input else None

    # Collect common inputs
    st.markdown("<hr>", unsafe_allow_html=True)
    # st.markdown("### Common Inputs")
    age = number_input_with_none("Enter your age")
    physically_active = st.selectbox(
        "How much physical activity do you get daily?",
        options=["", "Less than half an hour", "None", "More than half an hour", "One hour or more"]
    )
    bp_level = st.selectbox("What is your blood pressure level?", options=["", "High", "Normal", "Low"])
    high_bp = st.selectbox("Have you been diagnosed with high blood pressure?", options=["", "Yes", "No"])
    sleep = number_input_with_none("Average sleep time per day (in hours)")
    sound_sleep = number_input_with_none("Average hours of sound sleep")

    # Height and weight for BMI calculation
    # st.markdown("### Height and Weight for BMI")
    height_in = number_input_with_none("Height (in inches)")
    weight_lb = number_input_with_none("Weight (in pounds)")

    # Calculate and display BMI
    if height_in and weight_lb:
        bmi = (weight_lb * 703) / (height_in ** 2)
        st.success(f"Your calculated BMI is: **{bmi:.2f}**")
    else:
        st.warning("Please provide both height and weight for BMI calculation.")

    # Gender-specific inputs
    if st.session_state.gender == "Female":
        # st.markdown("### Female-Specific Questions")
        pregnancies = number_input_with_none("Number of pregnancies")
        gestation_history = st.selectbox("Have you had gestational diabetes?", options=["", "Yes", "No"])
        pcos = st.selectbox("Have you been diagnosed with PCOS?", options=["", "Yes", "No"])

        gender_specific_data = {
            'Pregnancies': pregnancies,
            'GestationHistory': gestation_history,
            'PCOS': pcos
        }

    elif st.session_state.gender == "Male":
        # st.markdown("### Male-Specific Questions")
        smoking = st.selectbox("Do you smoke?", options=["", "Yes", "No"])
        regular_medicine = st.selectbox("Do you take regular medicine?", options=["", "Yes", "No"])
        stress = st.selectbox("Do you experience high levels of stress?", options=["", "Yes", "No"])

        gender_specific_data = {
            'Smoking': smoking,
            'RegularMedicine': regular_medicine,
            'Stress': stress
        }

    # CGM Data Input Section
    st.markdown("### Continuous Glucose Monitoring (CGM) Data")
    time_series_input = st.text_area(
        "Enter your CGM data."
        "comma-separated, 20 values" 
        "enter in form 00:00:00 (hours:minutes:seconds since midnight): "
        "For 10 timesteps, 2 values per timestep"
    )

    # Combine all input data into a dictionary (just for storing purposes)
    input_data_dict = {
        'Age': age,
        'PhysicallyActive': physically_active,
        'BPLevel': bp_level,
        'highBP': high_bp,
        'Sleep': sleep,
        'SoundSleep': sound_sleep,
        'BMI': bmi if height_in and weight_lb else "Not calculated",
        'Gender': 1 if st.session_state.gender == "Male" else 0
    }
    input_data_dict.update(gender_specific_data)

    # Handle the "Submit" button
    if st.button("Submit"):
        if time_series_input:
            # Split the input string into a list of values
            data_list = [item.strip() for item in time_series_input.split(",")]
            # Check if the number of values is even
            if len(data_list) % 2 != 0:
                st.error("Please enter an even number of values for time and value pairs.")
            elif len(data_list) != 20:
                st.error("Please 20 values for time and value pairs.")
            else:
                # Create a list of tuples (time, value)
                timesteps = [(data_list[i], data_list[i + 1]) for i in range(0, len(data_list), 2)]
                # Create a DataFrame from the list of tuples
                df = pd.DataFrame(timesteps, columns=["Time", "Value"])
                # Define numerical columns
                numerical_columns = ['Time', 'Value']
                df[numerical_columns] = min_max_scaler.fit_transform(df[numerical_columns])
                flat_list = df.values.flatten().tolist()  # Convert to a flat list of values
                comma_separated_values = ",".join(map(str, flat_list))  # Join as a single string
                # Add this string to the input_data_dict
                input_data_dict['CGMData'] = comma_separated_values
        else:
            st.warning("Please provide CGM data.")

        st.write("Currently, no predictions are made as the model is not loaded.")
        st.info("Note: You would see predictions and diagnosis here if the model was loaded.")

        # # Saving to MongoDB would go here (commented out)
        # client = MongoClient(mongo_uri)
        # db = client['diabetes_data']
        # collection = db['predictions']
        # collection.insert_one({
        #     "input_values": input_data_dict,
        #     "prediction": "Placeholder",
        #     "diagnosis": "Placeholder"
        # })