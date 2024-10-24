import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import hashlib

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
diabetes_db = client['DiabetesRepo']
predictions_collection = diabetes_db['Diabetes_Prediction_Data']
user_db = client['Users']
credentials_collection = user_db['Credentials']

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if user exists in the database
def check_user_credentials(username, password):
    hashed_password = hash_password(password)
    user = credentials_collection.find_one({"username": username, "password": hashed_password})
    return user

# Sign-up function
def sign_up_user(username, password):
    hashed_password = hash_password(password)
    credentials_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "gender": None  # Gender will be added after login
    })

# Update gender in the database
def update_user_gender(username, gender):
    credentials_collection.update_one({"username": username}, {"$set": {"gender": gender}})

# Streamlit session state for managing login/signup
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
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

# Sign-up/Login Page
if not st.session_state.logged_in:
    styled_header("Diabetes Prediction App - Sign Up / Login")

    # Username and password input
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")

    # Check if both fields are filled before enabling buttons
    if username and password:
        if st.button("Sign Up"):
            # Check if user already exists
            if credentials_collection.find_one({"username": username}):
                st.warning("Username already exists. Please choose a different one.")
            else:
                # Sign up the user without gender (gender is selected after login)
                sign_up_user(username, password)
                st.success("Sign up successful! You can now log in.")

        if st.button("Log In"):
            # Validate login credentials
            user = check_user_credentials(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.gender = user['gender']
                if st.session_state.gender:
                    st.success(f"Welcome back, {username}!")
                else:
                    st.info(f"Please select your gender, {username}.")
                st.rerun()  # Refresh the app to load the next step
            else:
                st.error("Invalid username or password.")
    else:
        st.info("Please fill out both fields to enable sign-up and login.")

# Gender Selection Page (if gender not yet selected)
elif not st.session_state.gender:
    styled_header(f"Welcome {st.session_state.username}!")

    # Gender selection
    gender = st.selectbox("Select your gender", options=["Select your gender", "Male", "Female"])

    if gender != "Select your gender" and st.button("Submit"):
        st.session_state.gender = gender
        update_user_gender(st.session_state.username, gender)
        st.success(f"Gender selection successful! You can now proceed.")
        st.rerun()  # Refresh the app to load the prediction page

# Gender-Specific Prediction Page (once logged in and gender selected)
else:
    styled_header(f"Welcome {st.session_state.username}, Questionnaire for {st.session_state.gender} Patients")

    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.username = None
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

    # Gender-Specific Questions
    if st.session_state.gender == "Female":
        # (Prediction flow for females here, same as before...)
        pregnancies = st.number_input("Number of pregnancies", min_value=0, step=1)
        gestation_history = st.selectbox("Have you had gestational diabetes?", options=["", "Yes", "No"])
        pcos = st.selectbox("Have you been diagnosed with PCOS?", options=["", "Yes", "No"])
        # Add rest of the female-specific questions and logic...
        # Mock CGM input field for demonstration purposes
        cgm_input = st.text_area("Enter your CGM data (mock input), comma-separated, 20 values. Example: time1,value1,time2,value2,...")
        gender_specific_data = {'Pregnancies': pregnancies, 'Gestation in previous Pregnancy': gestation_history, 'PCOS': pcos}
        
    
    elif st.session_state.gender == "Male":
        # (Prediction flow for males here, same as before...)
        smoking = st.selectbox("Do you smoke?", options=["", "Yes", "No"])
        regular_medicine = st.selectbox("Do you take regular medicine for diabetes?", options=["", "Yes", "No"])
        stress = st.selectbox("Do you experience high levels of stress?", options=["", "Yes", "No"])
        # Add rest of the male-specific questions and logic...
        cgm_input = st.text_area("Enter your CGM data (mock input), comma-separated, 20 values. Example: time1,value1,time2,value2,...")
        gender_specific_data = {'Smoking': smoking, 'RegularMedicine': regular_medicine, 'Stress': stress}

    # The rest of the prediction code remains the same...
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

        # Prediction using the structured model
        if st.session_state.gender == "Female":
            # Define the expected feature names as they were during model training
            expected_feature_names = ['Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep', 'BPLevel', 'Pregnancies', 'Gestation in previous Pregnancy', 'PCOS']
            # Reorder the DataFrame to match the expected feature names
            input_data_df = input_data_df.reindex(columns=expected_feature_names)
            # Create the DMatrix
            d_matrix = xgb.DMatrix(data=input_data_df)
            structured_probs = female_structured_model.predict(d_matrix)
            predicted_class = np.argmax(structured_probs)
            st.success(f"The predicted class is: {class_labels[predicted_class]} with probability {np.max(structured_probs):.2f}")

            # Recommendations for female based on the predicted class
            if predicted_class == 0:  # No diabetes
                st.info(
                    "**Recommendation**: To reduce the risk of diabetes in the future: \n"
                    "- Maintain a balanced diet rich in fruits and vegetables. \n"
                    "- Engage in regular physical activity (at least 30 minutes daily). \n"
                    "- Monitor your weight and ensure a healthy BMI. \n"
                    "- Get regular health check-ups, especially if you have a family history of diabetes. \n"
                    "- If you had gestational diabetes during pregnancy, monitor blood sugar levels post-pregnancy as you may be at higher risk of developing type 2 diabetes."
                )
                # Additional feature-based recommendations
                if input_data_dict['PhysicallyActive'] in ["None", "Less than half an hour"]:
                    st.warning("Consider increasing your daily physical activity to at least 30 minutes to reduce the risk of diabetes.")
                if bmi and bmi >= 25:
                    st.warning("Your BMI indicates that you are overweight. Consider adopting a balanced diet and exercise plan to achieve a healthier BMI.")

            elif predicted_class == 3:  # Gestational diabetes
                st.info(
                    "**Recommendation**: Since you have been predicted with **gestational diabetes**: \n"
                    "- Follow your doctor’s advice closely to manage blood sugar levels during pregnancy. \n"
                    "- Maintain a healthy diet and engage in moderate physical activity. \n"
                    "- Post-pregnancy, continue monitoring your blood sugar levels as gestational diabetes can increase the risk of developing type 2 diabetes later in life."
                )

            else:  # Diabetes or prediabetes
                st.info(
                    "**Recommendation**: Since you have been predicted as diabetic or prediabetic: \n"
                    "- Consult a healthcare provider for personalized care. \n"
                    "- Regularly monitor your blood glucose levels. \n"
                    "- Follow a healthy eating plan recommended by a dietitian. \n"
                    "- Exercise regularly (at least 150 minutes of moderate activity per week). \n"
                    "- Take any prescribed medications on time. \n"
                    "- Consider regular screenings for heart health, as diabetes increases cardiovascular risks."
                )

        elif st.session_state.gender == "Male":
            # Define the expected feature names as they were during model training
            expected_feature_names = ['Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Smoking', 'Sleep', 'SoundSleep', 'RegularMedicine', 'Stress', 'BPLevel']
            # Reorder the DataFrame to match the expected feature names
            input_data_df = input_data_df.reindex(columns=expected_feature_names)
            # Create the DMatrix
            d_matrix = xgb.DMatrix(data=input_data_df)

            structured_probs = male_structured_model.predict(d_matrix)
            predicted_class = np.argmax(structured_probs)
            st.success(f"The predicted class is: {class_labels[predicted_class]} with probability {np.max(structured_probs):.2f}")
            
            # Recommendations for male based on the predicted class
            if predicted_class == 0:  # No diabetes
                st.info(
                    "**Recommendation**: To lower the risk of future diabetes: \n"
                    "- Incorporate regular physical activity into your daily routine (at least 30 minutes or more). \n"
                    "- Avoid smoking and excessive alcohol consumption. \n"
                    "- Eat a balanced diet and limit processed foods and sugars. \n"
                    "- Maintain a healthy weight and get regular health check-ups."
                )
                # Additional feature-based recommendations
                if input_data_dict['Smoking'] == "Yes":
                    st.warning("Smoking can increase the risk of diabetes. Consider quitting smoking for better health.")
                if bmi and bmi >= 25:
                    st.warning("Your BMI indicates you are overweight. A healthy BMI reduces the risk of diabetes.")
                if input_data_dict['PhysicallyActive'] in ["None", "Less than half an hour"]:
                    st.warning("Consider increasing your physical activity to at least 30 minutes daily to reduce the risk of diabetes.")

            else:  # Diabetes or prediabetes
                st.info(
                    "**Recommendation**: Based on your diagnosis of diabetes or prediabetes: \n"
                    "- Visit a healthcare professional for guidance. \n"
                    "- Keep track of your blood sugar levels regularly. \n"
                    "- Engage in regular physical activity (such as brisk walking or cycling). \n"
                    "- Follow your prescribed medications and treatment plan diligently. \n"
                    "- Consider adopting a diet low in refined sugars and saturated fats."
                )

        # Prepare the entry for MongoDB
        entry = {
            'username': st.session_state.username,
            **input_data_encoded,
            'class_probabilities': structured_probs.tolist(),  # Convert to list for JSON serialization
            'prediction': int(predicted_class),  # Ensure prediction is a standard integer
            'diagnosis': class_labels[predicted_class],
        }
        # Insert entry into MongoDB
        predictions_collection.insert_one(entry)
        st.success("Data successfully uploaded to MongoDB!")


