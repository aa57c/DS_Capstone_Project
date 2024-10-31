import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained models
female_structured_model = xgb.Booster()
female_structured_model.load_model('/Users/sreevardhanreddysoma/Desktop/HandsOn/xgboost_female.bin')
male_structured_model = xgb.Booster()
male_structured_model.load_model('/Users/sreevardhanreddysoma/Desktop/HandsOn/xgboost_male.bin')

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_DB_CONN_URL")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client['DiabetesRepo']
collection = db['Diabetes Prediction Data']

# Load BioClinicalBERT model for text analysis
@st.cache_resource
def load_nlp_model():
    model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return model, tokenizer

model, tokenizer = load_nlp_model()

# Hugging Face recommendation model
@st.cache_resource
def load_recommendation_model():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

recommendation_model = load_recommendation_model()

# Function to process medical text input
def process_medical_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.detach().numpy()[0][1]  # Return the probability of "risk"

# Function to generate recommendations using Hugging Face model based on risk factor
def generate_recommendations(risk_factor):
    prompt = f"The patient has a medical text risk factor of {risk_factor}. Provide recommendations for diabetes management, lifestyle changes, or medications based on this risk level."

    # Use the Hugging Face recommendation model
    response = recommendation_model(prompt, max_length=150, do_sample=True, temperature=0.7)
    return response[0]['generated_text']

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
        
    elif st.session_state.gender == "Male":
        smoking = st.selectbox("Do you smoke?", options=["", "Yes", "No"])
        regular_medicine = st.selectbox("Do you take regular medicine for diabetes?", options=["", "Yes", "No"])
        stress = st.selectbox("Do you experience high levels of stress?", options=["", "Yes", "No"])
        gender_specific_data = {'Smoking': smoking, 'RegularMedicine': regular_medicine, 'Stress': stress}

    # Medical history text input
    medical_text = st.text_area("Describe your medical history and symptoms (related to diabetes)")

    # Initialize risk factor to None
    risk_factor = None

    # Process medical text input using BioClinicalBERT if provided
    if medical_text:
        risk_factor = process_medical_text(medical_text, model, tokenizer)
        st.success(f"Risk factor extracted from medical text: **{risk_factor:.2f}**")

        # Generate recommendations based on risk factor using Hugging Face model
        recommendations = generate_recommendations(risk_factor)
        st.info(f"Recommendations: {recommendations}")
    else:
        st.warning("Please provide your medical history for better analysis.")

    input_data_dict = {
        'Age': age,
        'PhysicallyActive': physically_active,
        'BPLevel': bp_level,
        'HighBP': high_bp,
        'Sleep': sleep,
        'SoundSleep': sound_sleep,
        'BMI': bmi if height_in and weight_lb else None,
        'MedicalTextRiskFactor': risk_factor  # Include the processed risk factor
    }
    input_data_dict.update(gender_specific_data)

    if st.button("Submit"):
        # Ensure all required inputs are provided
        if None in input_data_dict.values():
            st.error("Please fill out all required fields before submitting.")
        else:
            # Encode categorical variables
            input_data_encoded = {}
            for key in input_data_dict.keys():
                if isinstance(input_data_dict[key], str) and input_data_dict[key]:
                    input_data_encoded[key] = custom_label_encode(input_data_dict[key], key)
                else:
                    input_data_encoded[key] = input_data_dict[key]  # Include numeric inputs as is

            st.warning(f"Encoded data: {input_data_encoded}")

            # Convert dictionary to DataFrame for model input
            df = pd.DataFrame([input_data_encoded])

            # Choose gender-specific model
            if st.session_state.gender == "Female":
                model = female_structured_model
            else:
                model = male_structured_model

            # Make predictions using XGBoost model
            dmatrix = xgb.DMatrix(df)
            prediction = model.predict(dmatrix)

            # Get class label from prediction
            predicted_class = np.argmax(prediction)
            st.success(f"Prediction: **{class_labels[predicted_class]}**")

            # Save the input data and prediction in MongoDB
            input_data_dict['Prediction'] = class_labels[predicted_class]
            collection.insert_one(input_data_dict)
            st.success("Your data has been saved successfully.")