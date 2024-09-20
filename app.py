import streamlit as st
import numpy as np
from pymongo import MongoClient
from keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

mongo_uri = os.getenv('MONGO_DB_CONN_URL')

# Load your combined model
model = load_model('final_deep_learning_model.h5')

# Assuming you saved the scaler used during training
# scaler = StandardScaler()

# Collect user inputs for fine-tuned features (15 features)
pregnancies = st.number_input("How many pregnancies have you had in total?", min_value=0, value=0)
# Collect user input for height and weight
height_in = st.number_input("Enter your height (in inches):", min_value=0.0, value=0.0)
weight_lb = st.number_input("Enter your weight (in pounds):", min_value=0.0, value=0.0)

# Calculate BMI
if height_in > 0 and weight_lb > 0:
    bmi = (weight_lb * 703) / (height_in ** 2)
    st.write(f"Your calculated BMI is: {bmi:.2f}")
else:
    st.write("Please enter valid height and weight to calculate BMI.")

sound_sleep = st.number_input("How many hours of sleep were you still in bed (sound sleep)?", value=0.0)
sleep = st.number_input("Enter total sleep on average per day in hours", value=0.0)

# Gender options
gender = st.selectbox("Select your gender", options=["Male", "Female"])
gender_male = 1 if gender == "Male" else 0
gender_female = 1 if gender == "Female" else 0

age = st.number_input("Enter age", value=0)

# Family diabetes history
family_diabetes = st.selectbox("Do you have a family history of diabetes?", options=["Yes", "No"])
family_diabetes_yes = 1 if family_diabetes == "Yes" else 0
family_diabetes_no = 1 if family_diabetes == "No" else 0

# Physical activity levels
physically_active = st.selectbox(
    "On average, how much are you physically active per day?", 
    options=["Less than half an hr", "None", "More than half an hr", "One hr or more"]
)
physically_active_less_than_half_hr = 1 if physically_active == "Less than half an hr" else 0
physically_active_none = 1 if physically_active == "None" else 0

# Blood pressure level
bp_level = st.selectbox(label="What is your blood pressure level?"
    "(less than 90/60 mmHg - Low, less than 120/80 mmHg - Normal, 140-159/90-99 - High)", options=["High", "Normal", "Low"])

bp_level_high = 1 if bp_level == "High" else 0
bp_level_normal = 1 if bp_level == "Normal" else 0

# High blood pressure status
high_bp = st.selectbox("Are you diagnosed with high blood pressure?", options=["Yes", "No"])
high_bp_yes = 1 if high_bp == "Yes" else 0
high_bp_no = 1 if high_bp == "No" else 0

# Collect user inputs for time-series CGM data (10 timesteps, 6 features per timestep)
time_series_input = st.text_area(
    "Enter your CGM (Continuous Glucose Monitoring) data: "
    "Please provide exactly 60 values, comma-separated, with 6 values per timestep for 10 timesteps."
    "Example: 100,105,110,120,115,100,..."
)

# Add a button to submit the features and make a prediction
if st.button("Submit"):
    if time_series_input:
        # Convert user input into numpy array and handle missing data
        time_series_values = np.array([float(x) for x in time_series_input.split(",")])

        # Ensure the CGM data matches the expected length (10 timesteps * 6 features = 60 values)
        if len(time_series_values) < 60:
            st.error("Please provide exactly 60 values (10 timesteps, 6 values per timestep).")
        else:
            # Reshape the time series input to match the model's input shape
            time_series_values = time_series_values.reshape(1, 10, 6)

            # Prepare fine-tuned input features (reshape to 1 sample, 15 features)
            fine_tuned_features = np.array([[pregnancies, bmi, sound_sleep, sleep,  
                                     gender_male, age, gender_female, family_diabetes_yes, 
                                     physically_active_less_than_half_hr, physically_active_none, 
                                     family_diabetes_no, bp_level_high, high_bp_no, high_bp_yes, 
                                     bp_level_normal]])
            
            # **Scale the features** before making the prediction
            # fine_tuned_features_scaled = scaler.transform(fine_tuned_features)
            # **Scale the time series data**
            # time_series_values_scaled = scaler.transform(time_series_values.reshape(-1, 6)).reshape(1, 10, 6)


            # Make the prediction
            prediction = model.predict([time_series_values, fine_tuned_features])
            # Get the index of the highest value in the prediction array (0, 1, 2, or 3)
            predicted_class = np.argmax(prediction)

            # Map the predicted class to a diagnosis
            diagnosis_map = {
                0: "No diabetes",
                1: "Prediabetes",
                2: "Type 2 diabetes",
                3: "Gestational diabetes"
            }

            # Get the corresponding diagnosis
            diagnosis = diagnosis_map[predicted_class]

            # Display the diagnosis
            st.write("Diagnosis:", diagnosis)

            # Upload the prediction to MongoDB
            client = MongoClient(mongo_uri)
            db = client['diabetes_data']
            collection = db['predictions']
            collection.insert_one({
                "input_values": {
                    "pregnancies": pregnancies,
                    "bmi": bmi,
                    "sound_sleep": sound_sleep,
                    "sleep": sleep,
                    "gender_male": gender_male,
                    "age": age,
                    "gender_female": gender_female,
                    "family_diabetes_yes": family_diabetes_yes,
                    "physically_active_less_than_half_hr": physically_active_less_than_half_hr,
                    "physically_active_none": physically_active_none,
                    "family_diabetes_no": family_diabetes_no,
                    "bp_level_high": bp_level_high,
                    "high_bp_no": high_bp_no,
                    "high_bp_yes": high_bp_yes,
                    "bp_level_normal": bp_level_normal,
                    "cgm_data": time_series_input,
                },
                "prediction": prediction.tolist(),
                "diagnosis": diagnosis
            })
    else:
        st.write("Please provide CGM data.")
