import streamlit as st
import numpy as np
import joblib
import xgboost as xgb
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load the pre-trained models
female_structured_model = xgb.Booster()
female_structured_model.load_model('xgboost_female.bin')
male_structured_model = xgb.Booster()
male_structured_model.load_model('xgboost_male.bin')

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_DB_CONN_URL")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client['DiabetesRepo']  # replace with your database name
collection = db['Diabetes Prediction Data']  # replace with your collection name

# Streamlit session state for gender selection
if 'gender' not in st.session_state:
    st.session_state.gender = None

# Helper function to add style to sections
def styled_header(title, subtitle=None):
    st.markdown(f"<h1 style='color: #4CAF50;'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<h3 style='color: #555;'>{subtitle}</h3>", unsafe_allow_html=True)

# Custom label encoding function
def custom_label_encode(value, key):
    encoding_dicts = {
        'BPLevel': {"Normal": 0, "Low": 1, "High": 2},
        'PhysicallyActive': {"None": 0, "Less than half an hour": 1, "More than half an hour": 2, "One hour or more": 3},
        'HighBP': {"No": 0, "Yes": 1},
        'Gestation in previous Pregnancy': {"No": 0, "Yes": 1},
        'PCOS': {"No": 0, "Yes": 1},
        'Smoking': {"No": 0, "Yes": 1},
        'RegularMedicine': {"No": 0, "Yes": 1},
        'Stress': {"No": 0, "Yes": 1}
    }
    return encoding_dicts.get(key, {}).get(value, value)

# Define class labels
class_labels = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Type 2 diabetes",
    3: "Gestational diabetes"
}

# Page 1: Gender Selection
if st.session_state.gender is None or st.session_state.gender == "Select your gender":
    styled_header("Diabetes Prediction App")
    st.session_state.gender = st.selectbox("Select your gender", options=["Select your gender", "Male", "Female"])
    if st.session_state.gender != "Select your gender":
        st.rerun()

# Gender-Specific Questions
else:
    styled_header(f"Questionnaire for {st.session_state.gender} Patients")
    st.markdown("Please fill out the details carefully. Accurate information helps in better prediction.")
    
    if st.button("Back to Gender Selection", key="back"):
        st.session_state.gender = None
        st.rerun()

    gender_specific_data = {}
    
    # Number input function
    def number_input_with_none(label):
        user_input = st.text_input(label)
        return float(user_input) if user_input else None

    age = number_input_with_none("Enter your age")
    physically_active = st.selectbox("How much physical activity do you get daily?", options=["", "Less than half an hour", "None", "More than half an hour", "One hour or more"])
    bp_level = st.selectbox("What is your blood pressure level?", options=["", "High", "Normal", "Low"])
    high_bp = st.selectbox("Have you been diagnosed with high blood pressure?", options=["", "Yes", "No"])
    sleep = number_input_with_none("Average sleep time per day (in hours)")
    sound_sleep = number_input_with_none("Average hours of sound sleep")
    height_in = number_input_with_none("Height (in inches)")
    weight_lb = number_input_with_none("Weight (in pounds)")

    if height_in and weight_lb:
        bmi = (weight_lb * 703) / (height_in ** 2)
        st.success(f"Your calculated BMI is: **{bmi:.2f}**")
    else:
        st.warning("Please provide both height and weight for BMI calculation.")

    if st.session_state.gender == "Female":
        pregnancies = number_input_with_none("Number of pregnancies")
        gestation_history = st.selectbox("Have you had gestational diabetes?", options=["", "Yes", "No"])
        pcos = st.selectbox("Have you been diagnosed with PCOS?", options=["", "Yes", "No"])
        gender_specific_data = {'Pregnancies': pregnancies, 'Gestation in previous Pregnancy': gestation_history, 'PCOS': pcos}
        
        # Mock CGM input field for demonstration purposes
        cgm_input = st.text_area("Enter your CGM data (mock input), comma-separated, 20 values. Example: time1,value1,time2,value2,...")

    elif st.session_state.gender == "Male":
        smoking = st.selectbox("Do you smoke?", options=["", "Yes", "No"])
        regular_medicine = st.selectbox("Do you take regular medicine for diabetes?", options=["", "Yes", "No"])
        stress = st.selectbox("Do you experience high levels of stress?", options=["", "Yes", "No"])
        gender_specific_data = {'Smoking': smoking, 'RegularMedicine': regular_medicine, 'Stress': stress}
        # Mock CGM input field for demonstration purposes
        cgm_input = st.text_area("Enter your CGM data (mock input), comma-separated, 20 values. Example: time1,value1,time2,value2,...")

    input_data_dict = {
        'Age': age,
        'PhysicallyActive': physically_active,
        'BPLevel': bp_level,
        'HighBP': high_bp,
        'Sleep': sleep,
        'SoundSleep': sound_sleep,
        'BMI': bmi if height_in and weight_lb else None
    }
    input_data_dict.update(gender_specific_data)

    if st.button("Submit"):
        # Create a new dictionary for encoded data excluding 'Gender'
        input_data_encoded = {}

        # Encode categorical variables
        for key in input_data_dict.keys():
            if isinstance(input_data_dict[key], str) and input_data_dict[key]:
                input_data_encoded[key] = custom_label_encode(input_data_dict[key], key)
            else:
                input_data_encoded[key] = input_data_dict[key]  # Include numeric inputs as is

        st.warning(f"Encoded categorical data: {input_data_encoded}")

        # Display mock CGM input if provided
        if st.session_state.gender == "Female" and cgm_input:
            st.info(f"Mock CGM Data Received: {cgm_input}")
        
        # Convert to DataFrame for prediction
        input_data_df = pd.DataFrame([input_data_encoded])  # Create DataFrame from dictionary

        # Define the expected feature names as they were during model training
        expected_feature_names = ['Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep', 'BPLevel', 'Pregnancies', 'Gestation in previous Pregnancy', 'PCOS']
        
        # Reorder the DataFrame to match the expected feature names
        input_data_df = input_data_df.reindex(columns=expected_feature_names)
        # Create the DMatrix
        d_matrix = xgb.DMatrix(data=input_data_df)

        # Prediction using the structured model
        if st.session_state.gender == "Female":
            structured_probs = female_structured_model.predict(d_matrix)
            predicted_class = np.argmax(structured_probs)
            st.success(f"The predicted class is: {class_labels[predicted_class]} with probability {np.max(structured_probs):.2f}")

        elif st.session_state.gender == "Male":
            structured_probs = male_structured_model.predict(d_matrix)
            predicted_class = np.argmax(structured_probs)
            st.success(f"The predicted class is: {class_labels[predicted_class]} with probability {np.max(structured_probs):.2f}")
        
        # Prepare the entry for MongoDB
        entry = {
            **input_data_encoded,
            'Gender': st.session_state.gender,
            'class_probabilities': structured_probs.tolist(),  # Convert to list for JSON serialization
            'prediction': int(predicted_class),  # Ensure prediction is a standard integer
            'diagnosis': class_labels[predicted_class]
        }
        # Insert entry into MongoDB
        collection.insert_one(entry)
        st.success("Data successfully uploaded to MongoDB!")

