import streamlit as st
import numpy as np
import xgboost as xgb
import tensorflow as tf
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import hashlib
import datetime
import joblib
import time
import requests
# Load pre-trained models


female_model = xgb.Booster()
female_model.load_model('xgboost_female.json')

male_model = xgb.Booster()
male_model.load_model('xgboost_male.json')

cgm_model = tf.keras.models.load_model('cgm_model.keras')
scaler = joblib.load('minmax_scaler.pkl')


load_dotenv()
mongo_uri = os.getenv("MONGO_DB_CONN_URL")
client = MongoClient(mongo_uri)
db = client['DiabetesRepo']
predictions_collection = db['Diabetes_Prediction_Data']
credentials_collection = client['Users']['Credentials']

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if user exists in the database
@st.cache_resource
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

API_URL = 'http://localhost:11434/api/generate'
def generate_recommendations(user_data):
    payload = {
        "model": "llama3.2",
        "prompt": f"Give recommendations for someone with these characteristics: {user_data}",
        "stream": False
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        # Convert the response content from JSON and print it
        response_json = response.json()
        return response_json['response']
    else:
        st.error(f"Error: {response.status_code}")
        return []




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
                    time.sleep(5)
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
        predictions_collection.insert_one({'username': st.session_state.username, 'data': []})
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

    
    # Number input function
    def number_input_with_none(label):
        user_input = st.text_input(label)
        return float(user_input) if user_input else None
    
    input_data_dict = {}

    binary_yes_no_options = {
        "Yes": 1,
        "No" : 0
    }


    age = number_input_with_none("Enter your age")

    st.write("Have you been diagnosed with high blood pressure?")
    selected_high_bp = st.radio(
        "Select your option:",
        options=list(binary_yes_no_options.keys()),
        key="high_bp_key"
    )
    # Retrieve the encoded value for the selected option
    high_bp = binary_yes_no_options[selected_high_bp]

    st.write("How many days per week are you typically physically active? Please select the option that best describes your activity level.")
    physical_activity_options = {
    "Not Active (Rarely or never active during the week)": 0,
    "Lightly Active (1-2 days per week with light physical activity)": 1,
    "Moderately Active (3-4 days per week, moderate activities like brisk walking)": 2,
    "Very Active (5 or more days per week, vigorous activities like running)": 3
    }
    # Create a radio button for activity level selection
    selected_physical_activity = st.radio(
        "Select your physical activity level per week:",
        options=list(physical_activity_options.keys())
    )

    physicallyactive = physical_activity_options[selected_physical_activity]

    height_in = number_input_with_none("Height (in inches)")
    weight_lb = number_input_with_none("Weight (in pounds)")
    if height_in and weight_lb:
        bmi = (weight_lb * 703) / (height_in ** 2)
        st.success(f"Your calculated BMI is: **{bmi:.2f}**")
    else:
        st.warning("Please provide both height and weight for BMI calculation.")
    
    sleep = number_input_with_none("Average sleep time per day (in hours)")
    sound_sleep = number_input_with_none("Average hours of sound sleep (sleep when you are lying completely still)")

    st.write("How often do you eat junk food (foods high in sugar and cholesterol) per week?")
    junk_food_options = {
        "Occasionally": 0,
        "Often": 1,
        "Very Often": 2,
        "Always": 3
    }
    selected_junk_food = st.radio(
        "Select how often you eat junk food:",
        options=list(junk_food_options.keys())
    )

    junkfood = junk_food_options[selected_junk_food]

    st.write("What is your blood pressure level?")
    bp_level_options = {
        "Normal": 0,
        "Low": 1,
        "High": 2
    }
    selected_bp_level = st.radio(
        "Select your blood pressure level:",
        options=list(bp_level_options.keys())
    )
    bp_level = bp_level_options[selected_bp_level]

    st.write("How often do you have to urinate per day?")
    urination_freq_options = {
        "Roughly 4 to 7 times per day": 0,
        "More than 7 to 10 times per day": 1,
    }

    selected_urination_freq = st.radio(
        "Select how frequently you urinate per day:",
        options=list(urination_freq_options.keys())
    )

    urinationfreq = urination_freq_options[selected_urination_freq]

    st.write("Are you diagnosed with high cholesterol?")
    selected_high_chol_option = st.radio(
        "Select your option:",
        options=list(binary_yes_no_options.keys()),
        key="high_chol_key"
    )
    high_chol = binary_yes_no_options[selected_high_chol_option]

    st.write("Do you consume fruit per day?")
    selected_fruit_option = st.radio(
        "Select your option:",
        options=list(binary_yes_no_options.keys()),
        key="fruits_key"
    )
    st.write("Do you consume vegetables per day?")
    selected_veggies_option = st.radio(
        "Select your option:",
        options=list(binary_yes_no_options.keys()),
        key="veggies_key"
    )

    fruits = binary_yes_no_options[selected_fruit_option]
    veggies = binary_yes_no_options[selected_veggies_option]

    gen_hlth_options = {
        "Excellent": 1,
        "Very Good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5
    }
    st.write("How would you describe your general health?")
    selected_gen_hlth_option = st.radio(
        "Would you say that in general your health is:",
        options=list(gen_hlth_options.keys()),
        key="gen_hlth_key"
    )
    gen_hlth = gen_hlth_options[selected_gen_hlth_option]

    phys_hlth = number_input_with_none("Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?")

    st.write("Have you experienced sudden loss of weight? (a loss of more than 5 percent of your body weight)")
    selected_weight_loss_option = st.radio(
        "Select your option:",
        options=list(binary_yes_no_options.keys()),
        key='weight_loss_key'
    )

    sudden_weight_loss = binary_yes_no_options[selected_weight_loss_option]

    st.write("Have you experienced any blurred vision this week?")
    selected_visual_blur_option = st.radio(
        "Select your option:",
        options=list(binary_yes_no_options.keys()),
        key="blurred_vision_key"
    )
    visual_blurring = binary_yes_no_options[selected_visual_blur_option]

    st.write("If you got injured, did you notice if your wound was healing slowly?")
    selected_healing_option = st.radio(
        "Select your option:",
        options=list(binary_yes_no_options.keys()),
        key="delayed_healing_key"   
    )
    delayed_healing = binary_yes_no_options[selected_healing_option]

    # Gender-Specific Questions
    if st.session_state.gender == "Female":
        # (Prediction flow for females here, same as before...)
        pregnancies = st.number_input("How many pregnancies have you had?", min_value=0, step=1)
        st.write("Have you had gestational diabetes before in those pregnancies?")
        selected_gestational_hist_option = st.radio(
            "Select your option:",
            options=list(binary_yes_no_options.keys()),
            key="gestation_hist_key"      
        )
        gestation_history = binary_yes_no_options[selected_gestational_hist_option]


        st.write("Are you currently pregnant?")
        selected_pregnant_option = st.radio(
            "Select your option:",
            options=list(binary_yes_no_options.keys()),
            key="pregnant_key"         
        )
        pregnant = binary_yes_no_options[selected_pregnant_option]

        st.write("Have you been diagnosed with PCOS?")
        selected_pcos_option = st.radio(
            "Select your option:",
            options=list(binary_yes_no_options.keys()),
            key="pcos_key"   
        )
        pcos = binary_yes_no_options[selected_pcos_option]

        # Add rest of the female-specific questions and logic...
        # Mock CGM input field for demonstration purposes
        # cgm_input = st.text_area("Enter your CGM data (mock input), comma-separated, 20 values. Example: time1,value1,time2,value2,...")

        input_data_dict = {
            'Age': age,
            'HighBP': high_bp,
            'PhysicallyActive': physicallyactive,
            'BMI': bmi if height_in and weight_lb else None,
            'Sleep': sleep,
            'SoundSleep': sound_sleep,
            'JunkFood': junkfood,
            'BPLevel': bp_level,
            'Pregnancies': pregnancies,
            'UriationFreq': urinationfreq,
            'HighChol': high_chol,
            "Fruits": fruits,
            "Veggies": veggies,
            "GenHlth": gen_hlth,
            "PhysHlth": phys_hlth,
            "Gestation in previous pregnancy": gestation_history,
            "PCOS": pcos,
            "sudden weight loss": sudden_weight_loss,
            "visual blurring": visual_blurring,
            "delayed healing": delayed_healing,
            "Pregnant": pregnant
        }
    elif st.session_state.gender == "Male":
        input_data_dict = {
            'Age': age,
            'HighBP': high_bp,
            'PhysicallyActive': physicallyactive,
            'BMI': bmi if height_in and weight_lb else None,
            'Sleep': sleep,
            'SoundSleep': sound_sleep,
            'JunkFood': junkfood,
            'BPLevel': bp_level,
            'UriationFreq': urinationfreq,
            'HighChol': high_chol,
            "Fruits": fruits,
            "Veggies": veggies,
            "GenHlth": gen_hlth,
            "PhysHlth": phys_hlth,
            "sudden weight loss": sudden_weight_loss,
            "visual blurring": visual_blurring,
            "delayed healing": delayed_healing,
        }
    cgm_input = st.text_area("Enter your CGM data, comma-separated, 24 values for each hour of the day. Example: glucose_value1,glucose_value2,glucose_value3,...")
    
    if st.button("Submit"):
        # Parse and process the CGM input values
        cgm_values = cgm_input.split(",")  # Split comma-separated input
        try:
            cgm_values = [float(val.strip()) for val in cgm_values if val.strip()]  # Convert to floats and remove any whitespace
            assert len(cgm_values) == 24, "Please enter exactly 24 values for CGM data."  # Ensure there are exactly 24 values
        except ValueError:
            st.error("Invalid CGM data format. Please enter numeric values only.")
        except AssertionError as e:
            st.error(e)
        # Convert to DataFrame for prediction
        input_data_df = pd.DataFrame([input_data_dict])  # Create DataFrame from dictionary
        cgm_scaled = scaler.transform(np.array(cgm_values).reshape(-1, 1)).flatten()  # Scale and flatten the array
        # Prepare CGM data for the LSTM model
        cgm_lstm_input = np.array(cgm_scaled).reshape((1, 24, 1))  # Shape to (1, 24, 1)
        lstm_prediction = cgm_model.predict(cgm_lstm_input)
        combined_preds = None
        structured_probs = None
        # Prediction using the structured model
        if st.session_state.gender == "Female":
            # Define the expected feature names as they were during model training
            expected_feature_names = ['Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep',
                                      'JunkFood', 'BPLevel', 'Pregnancies', 'UriationFreq', 'HighChol',
                                      'Fruits', 'Veggies', 'GenHlth', 'PhysHlth',
                                      'Gestation in previous Pregnancy', 'PCOS', 'sudden weight loss',
                                      'visual blurring', 'delayed healing', 'Pregnant']
            
            # Reorder the DataFrame to match the expected feature names
            input_data_df = input_data_df.reindex(columns=expected_feature_names)
            # Create the DMatrix
            d_matrix = xgb.DMatrix(data=input_data_df)
            structured_probs = female_model.predict(d_matrix)
            combined_preds = (lstm_prediction + structured_probs) / 2

        elif st.session_state.gender == "Male":
            # Define the expected feature names as they were during model training
            expected_feature_names = ['Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep', 'JunkFood', 'BPLevel', 'UriationFreq', 'HighChol', 'Fruits', 'Veggies', 'GenHlth', 'PhysHlth', 'sudden weight loss', 'visual blurring', 'delayed healing']
            # Reorder the DataFrame to match the expected feature names
            input_data_df = input_data_df.reindex(columns=expected_feature_names)
            # Create the DMatrix
            d_matrix = xgb.DMatrix(data=input_data_df)
            structured_probs = male_model.predict(d_matrix)
            lstm_preds_male = lstm_prediction[:, :3]  # Ignore the 4th class (gestational)
            # Combine the predictions. Here you might want to average the probabilities or take a majority vote
            combined_preds = (lstm_preds_male + structured_probs) / 2  # Averaging, adjust as needed

        predicted_class = np.argmax(combined_preds)
        st.success(f"The predicted class is: {class_labels[predicted_class]} with probability {np.max(structured_probs):.2f}")
        # Add timestamp to the input data dictionary
        input_data_dict['timestamp'] = datetime.datetime.now()
        input_data_dict['gender'] = st.session_state.gender
        input_data_dict['cgm'] = cgm_lstm_input.tolist()
        # Generating a user input summary for recommendations
        user_input_summary = ", ".join([f"{k}: {v}" for k, v in input_data_dict.items()])
        # Show a spinner and waiting message while generating recommendations
        with st.spinner("Getting recommendations..."):
            recommendations = generate_recommendations(user_input_summary)

        st.info(recommendations)

        # Prepare the entry for MongoDB
        query = {'username': st.session_state.username}
        new_value = {**input_data_dict, 'class_probabilities': combined_preds.tolist(),  # Convert to list for JSON serialization
        'prediction': int(predicted_class),  # Ensure prediction is a standard integer
        'diagnosis': class_labels[predicted_class], 'recommendations': recommendations}
        update = {'$push': {'data': new_value}}
        # Insert entry into MongoDB
        predictions_collection.update_one(query, update)
        st.success(f"Data successfully updated for {st.session_state.username}")




