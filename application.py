# Imports
import streamlit as st
import numpy as np
import pandas as pd
import os
import hashlib
import datetime
import requests
from pymongo import MongoClient
import joblib
import xgboost as xgb
import tensorflow as tf
import boto3

# Load environment variables

MONGO_URI = st.secrets["mongo_db"]["mongo_db_conn_url"]
BUCKET_NAME = st.secrets["s3"]["bucket_name"]
# API Call Function
#API_URL = 'http://localhost:11434/api/generate'

# Define class labels
CLASS_LABELS = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Type 2 diabetes",
    3: "Gestational diabetes"
}

# Define the expected feature names for female and male patients
female_feature_names = [
    'Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep', 'JunkFood', 'BPLevel', 
    'Pregnancies', 'UriationFreq', 'HighChol', 'Fruits', 'Veggies', 'GenHlth', 'PhysHlth', 
    'Gestation in previous Pregnancy', 'PCOS', 'sudden weight loss', 'visual blurring', 
    'delayed healing', 'Pregnant'
]

male_feature_names = [
    'Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep', 'JunkFood', 'BPLevel', 
    'UriationFreq', 'HighChol', 'Fruits', 'Veggies', 'GenHlth', 'PhysHlth', 
    'sudden weight loss', 'visual blurring', 'delayed healing'
]

# Define options
binary_yes_no_options = {"Yes": 1, "No": 0}
physical_activity_options = {
    "Not Active": 0, "Lightly Active": 1, "Moderately Active": 2, "Very Active": 3
}
junk_food_options = {"Occasionally": 0, "Often": 1, "Very Often": 2, "Always": 3}
bp_level_options = {"Normal": 0, "Low": 1, "High": 2}
urination_freq_options = {"4-7 times/day": 0, "More than 7-10 times/day": 1}
gen_hlth_options = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
 

# Initialize MongoDB client and database collections
@st.cache_resource
def get_mongo_collections():
    client = MongoClient(MONGO_URI)
    diabetes_db = client['DiabetesRepo']
    user_db = client['Users']
    predictions_collection = diabetes_db['Diabetes_Prediction_Data']
    credentials_collection = user_db['Credentials']
    return predictions_collection, credentials_collection


# Download pre-trained models from AWS S3
def download_models_from_s3(bucket, key, filename):
    s3 = boto3.client(
    's3',
    aws_access_key_id=st.secrets["s3"]["access_key"],
    aws_secret_access_key=st.secrets["s3"]["secret_key"])
    s3.download_file(bucket, key, filename)

@st.cache_resource
def load_models():
    female_model = xgb.Booster()
    female_model.load_model(f'/tmp/xgboost_female.json')

    male_model = xgb.Booster()
    male_model.load_model('/tmp/xgboost_male.json')

    cgm_model = tf.keras.models.load_model('/tmp/cgm_model.keras')
    scaler = joblib.load('/tmp/minmax_scaler.pkl')
    return female_model, male_model, cgm_model, scaler

models_to_load = ['xgboost_female.json', 'xgboost_male.json', 'cgm_model.keras', 'minmax_scaler.pkl']

for model in models_to_load:
    download_models_from_s3(BUCKET_NAME, model, f'/tmp/{model}')

female_model, male_model, cgm_model, scaler = load_models()
predictions_collection, credentials_collection = get_mongo_collections()

# Utility Functions
def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_resource
def check_user_credentials(username, password):
    """Check if user credentials are valid."""
    hashed_password = hash_password(password)
    user = credentials_collection.find_one({"username": username, "password": hashed_password})
    return user

def sign_up_user(username, password):
    """Create a new user."""
    hashed_password = hash_password(password)
    
    credentials_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "gender": None  # Gender will be added after login
    })
    

def update_user_gender(username, gender):
    """Update user's gender information."""
    credentials_collection.update_one({"username": username}, {"$set": {"gender": gender}})


#def generate_recommendations(user_data):
 #   Fetch recommendations based on user data.
  #  payload = {
   #     "model": "llama3.2",
    #    "prompt": f"Provide a personalized lifestyle and dietary recommendation based on the following characteristics: {user_data}. Do not provide medical advice, just general wellness recommendations.",
     #   "stream": False
    #}
    #response = requests.post(API_URL, json=payload)
    #if response.status_code == 200:
     #   response_json = response.json()
      #  return response_json.get('response', [])
    #else:
     #   st.error(f"Error: {response.status_code}")
      #  return []


# UI Helper Functions
def styled_header(title, subtitle=None):
    """Display a styled header."""
    st.markdown(f"<h1 style='color: #4CAF50;'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<h3 style='color: #555;'>{subtitle}</h3>", unsafe_allow_html=True)


# Streamlit session state initialization
def initialize_session_state():
    """Initialize session state variables if they are not already set."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'gender' not in st.session_state:
        st.session_state.gender = None

# Call the initialization function at the start of the app
initialize_session_state()


# Sign-up/Login and Gender Selection Functions
def display_login_page():
    """Display the login and sign-up page."""
    styled_header("Diabetes Prediction App - Sign Up / Login")

    # Username and password input
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")

    # Check if both fields are filled before enabling buttons
    if username and password:
        if st.button("Sign Up"):
            handle_signup(username, password)

        if st.button("Log In"):
            handle_login(username, password)
    else:
        st.info("Please fill out both fields to enable sign-up and login.")

def handle_signup(username, password):
    """Handle user sign-up."""
    if credentials_collection.find_one({"username": username}):
        st.warning("Username already exists. Please choose a different one.")
    else:
        sign_up_user(username, password)
        st.success("Sign up successful! You can now log in.")

def handle_login(username, password):
    """Handle user login."""
    user = check_user_credentials(username, password)
    if user:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.gender = user['gender']
        welcome_user(user)
        st.rerun()  # Refresh the app to load the next step
    else:
        st.error("Invalid username or password.")

def welcome_user(user):
    """Display a welcome message based on the user's gender."""
    if user['gender']:
        st.success(f"Welcome back, {user['username']}!")
    else:
        st.info(f"Please select your gender, {user['username']}.")

def display_gender_selection():
    """Display the gender selection page if not already set."""
    styled_header(f"Welcome {st.session_state.username}!")

    # Gender selection
    gender = st.selectbox("Select your gender", options=["Select your gender", "Male", "Female"])

    if gender != "Select your gender" and st.button("Submit"):
        st.session_state.gender = gender
        update_user_gender(st.session_state.username, gender)
        initialize_user_data(st.session_state.username)
        st.success("Gender selection successful! You can now proceed.")
        st.rerun()  # Refresh the app to load the prediction page

def initialize_user_data(username):
    """Initialize a new user's data entry in the predictions collection."""
    predictions_collection.insert_one({'username': username, 'data': []})

# Helper Functions
def logout():
    """Log the user out and reset session state."""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.gender = None
    st.rerun()

def number_input_with_none(label):
    user_input = st.text_input(label)
    return float(user_input) if user_input else None
def calculate_bmi():
    """Calculate and display BMI from height and weight."""
    height_in = number_input_with_none("Height (in inches)")
    weight_lb = number_input_with_none("Weight (in pounds)")
    if height_in and weight_lb:
        bmi = (weight_lb * 703) / (height_in ** 2)
        st.success(f"Your calculated BMI is: **{bmi:.2f}**")
        return bmi
    st.warning("Provide both height and weight for BMI calculation.")
    return None

def collect_user_inputs():
    """Collects common user inputs and returns as a dictionary."""
    input_data = {
        'Age': number_input_with_none("Enter your age"),
        'HighBP': st.radio("Have you been diagnosed with high blood pressure?", list(binary_yes_no_options.keys()), key="high_bp_key"),
        'PhysicallyActive': st.radio("Physical activity per week:", list(physical_activity_options.keys())),
        'BMI': calculate_bmi(),
        'Sleep': number_input_with_none("Average sleep time per day (in hours)"),
        'SoundSleep': number_input_with_none("Average hours of sound sleep"),
        'JunkFood': st.radio("How often do you eat junk food?", list(junk_food_options.keys())),
        'BPLevel': st.radio("Blood pressure level:", list(bp_level_options.keys())),
        'UriationFreq': st.radio("Frequency of urination:", list(urination_freq_options.keys())),
        'HighChol': st.radio("Diagnosed with high cholesterol?", list(binary_yes_no_options.keys()), key="high_chol_key"),
        'Fruits': st.radio("Consume fruit daily?", list(binary_yes_no_options.keys()), key="fruits_key"),
        'Veggies': st.radio("Consume vegetables daily?", list(binary_yes_no_options.keys()), key="veggies_key"),
        'GenHlth': st.radio("General health status:", list(gen_hlth_options.keys()), key="gen_hlth_key"),
        'PhysHlth': number_input_with_none("Physical health (days not good in the last 30 days)"),
        'sudden weight loss': st.radio("Experienced sudden weight loss?", list(binary_yes_no_options.keys()), key="weight_loss_key"),
        'visual blurring': st.radio("Experienced blurred vision?", list(binary_yes_no_options.keys()), key="blurred_vision_key"),
        'delayed healing': st.radio("Wounds heal slowly?", list(binary_yes_no_options.keys()), key="delayed_healing_key"),
    }
    return input_data

def collect_female_specific_inputs():
    """Collects inputs specific to female users."""
    pregnancies = st.number_input("Number of pregnancies", min_value=0, step=1)
    gestation_history = st.radio("Had gestational diabetes in pregnancies?", list(binary_yes_no_options.keys()), key="gestation_hist_key") if pregnancies > 0 else 0
    pregnant = st.radio("Currently pregnant?", list(binary_yes_no_options.keys()), key="pregnant_key")
    pcos = st.radio("Diagnosed with PCOS?", list(binary_yes_no_options.keys()), key="pcos_key")

    return {
        "Pregnancies": pregnancies,
        "Gestation in previous pregnancy": gestation_history,
        "Pregnant": pregnant,
        "PCOS": pcos
    }

def collect_cgm_input():
    """Collect and validate CGM input."""
    cgm_input = st.text_area("Enter your CGM data, comma-separated, 24 values for each hour of the day.")
    cgm_values = cgm_input.split(",")
    try:
        cgm_values = [float(val.strip()) for val in cgm_values if val.strip()]
        assert len(cgm_values) == 24, "Enter exactly 24 values for CGM data."
        return cgm_values
    except (ValueError, AssertionError) as e:
        st.error(e)
        return None

def flatten_cgm_data(nested_cgm):
    # Flatten the nested list
    flattened_cgm = [item[0] for sublist in nested_cgm for item in sublist]
    return flattened_cgm

def save_to_mongodb(input_data_dict, combined_preds, predicted_class, cgm_lstm_input):
    
   # """Saves user input and prediction results to MongoDB."""
    # Prepare a summary for recommendations
    #user_input_summary = ", ".join([
     #   f"{k}: {'Yes' if v == 1 else 'No' if v == 0 else v}" 
      #  for k, v in input_data_dict.items()
    #])

    # Generate recommendations based on user input summary
    #with st.spinner("Getting recommendations..."):
     #   recommendations = generate_recommendations(user_input_summary)

    # Display the recommendations
    #st.info(recommendations)
    

    input_data_dict.update({
        'timestamp': datetime.datetime.now(),
        'gender': st.session_state.gender,
        'class_probabilities': combined_preds.tolist(),
        'prediction': int(predicted_class),
        'diagnosis': CLASS_LABELS[predicted_class],
        'recommendations': recommendations,
        'cgm': flatten_cgm_data(cgm_lstm_input)
    })
    predictions_collection.update_one(
        {'username': st.session_state.username},
        {'$push': {'data': input_data_dict}}
    )
    st.success(f"Data successfully updated for {st.session_state.username}")
def xgboost_predict(input_data_df):
    """Runs prediction using XGBoost model."""
    if st.session_state.gender == "Female":
        input_data_df = input_data_df.reindex(columns=female_feature_names)
        d_matrix = xgb.DMatrix(data=input_data_df)
        return female_model.predict(d_matrix)
    else:
        input_data_df = input_data_df.reindex(columns=male_feature_names)
        d_matrix = xgb.DMatrix(data=input_data_df)
        return male_model.predict(d_matrix)

def predict(input_data_dict, cgm_values):
    """Runs the predictions for structured data and CGM data."""
    # Create DataFrame and reshape CGM data for prediction
    input_data_df = pd.DataFrame([input_data_dict])
    cgm_scaled = scaler.transform(np.array(cgm_values).reshape(-1, 1)).flatten()
    cgm_lstm_input = np.array(cgm_scaled).reshape((1, 24, 1))

    # Run LSTM and structured model predictions
    lstm_prediction = cgm_model.predict(cgm_lstm_input)

    # Ignore the last class if session_state equals "Male"
    if st.session_state.get('gender') == "Male":
        lstm_prediction = lstm_prediction[:, :-1]  # Remove the last class

    structured_probs = xgboost_predict(input_data_df)
    combined_preds = (lstm_prediction + structured_probs) / 2
    predicted_class = np.argmax(combined_preds)
    
    return structured_probs, combined_preds, predicted_class, cgm_lstm_input


def process_and_submit(input_data_dict, cgm_values):
    """Processes inputs, runs predictions, and updates MongoDB."""
    if cgm_values is None:
        return  # Exit if CGM data is invalid
    
    # Prepare and predict
    structured_probs, combined_preds, predicted_class, cgm_lstm_input = predict(input_data_dict, cgm_values)

    # Display results
    st.success(f"Predicted class: {CLASS_LABELS[predicted_class]} with probability {np.max(structured_probs):.2f}")
    
    # Add data to MongoDB
    save_to_mongodb(input_data_dict, combined_preds, predicted_class, cgm_lstm_input)

def encode_inputs(input_data):
    # Now map the options to their values
    for key, value in input_data.items():
        if value in binary_yes_no_options.keys():
            input_data[key] = binary_yes_no_options[input_data[key]]
        elif value in physical_activity_options.keys():
            input_data[key] = physical_activity_options[input_data[key]]
        elif value in junk_food_options.keys():
            input_data[key] = junk_food_options[input_data[key]]
        elif value in bp_level_options.keys():
            input_data[key] = bp_level_options[input_data[key]]
        elif value in urination_freq_options.keys():
            input_data[key] = urination_freq_options[input_data[key]]
        elif value in gen_hlth_options.keys():
            input_data[key] = gen_hlth_options[input_data[key]]
    return input_data



# Main Logic for Sign-up/Login or Gender Selection
if not st.session_state.logged_in:
    display_login_page()
elif not st.session_state.gender:
    display_gender_selection()

# Gender-Specific Prediction Page (once logged in and gender selected)
else:
    styled_header(f"Welcome {st.session_state.username}, Questionnaire for {st.session_state.gender} Patients")

    if st.button("Log Out"):
        logout()


    # Collect user input
    input_data_dict = collect_user_inputs()

    # Gender-Specific Questions
    if st.session_state.gender == "Female":
        input_data_dict.update(collect_female_specific_inputs())

    input_data_dict = encode_inputs(input_data_dict)
    
    cgm_values = collect_cgm_input()

    if st.button("Submit"):
        process_and_submit(input_data_dict, cgm_values)
