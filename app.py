import streamlit as st  # Streamlit is used for building web applications easily with Python
import numpy as np  # Numpy is used for numerical operations, mainly handling arrays in this script
from pymongo import MongoClient  # MongoDB client for connecting and interacting with the database
from keras.models import load_model  # type: ignore # For loading the pre-trained Keras deep learning model
from sklearn.preprocessing import StandardScaler  # StandardScaler for feature scaling during model input

from dotenv import load_dotenv  # Load environment variables from a .env file for secure configuration
import os  # For interacting with operating system functionalities, such as accessing environment variables

# Load environment variables from the .env file (useful for sensitive data like database connection strings)
load_dotenv()

# Get MongoDB URI from environment variables for secure connection to the MongoDB database
mongo_uri = os.getenv('MONGO_DB_CONN_URL')

# Load the pre-trained deep learning model (assumed to be a .h5 Keras model)
model = load_model('final_deep_learning_model.h5')

# For testing purposes: Initially, scaling wasn't working properly so the scaler is commented out
# scaler = StandardScaler()

# Collect user inputs using Streamlit's user interface components
# The input corresponds to the number of pregnancies the user has had
pregnancies = st.number_input("How many pregnancies have you had in total?", min_value=0, value=0)

# Collect user input for height in inches and weight in pounds
height_in = st.number_input("Enter your height (in inches):", min_value=0.0, value=0.0)
weight_lb = st.number_input("Enter your weight (in pounds):", min_value=0.0, value=0.0)

# Calculate the user's BMI using the provided height and weight values
if height_in > 0 and weight_lb > 0:
    # BMI calculation formula for imperial units: (weight in pounds * 703) / (height in inches)^2
    bmi = (weight_lb * 703) / (height_in ** 2)
    st.write(f"Your calculated BMI is: {bmi:.2f}")
else:
    st.write("Please enter valid height and weight to calculate BMI.")

# Collect more user inputs related to health: Sound sleep and total sleep in hours
sound_sleep = st.number_input("How many hours of sleep were you still in bed (sound sleep)?", value=0.0)
sleep = st.number_input("Enter total sleep on average per day in hours", value=0.0)

# Gender selection input (either Male or Female), with corresponding binary encoding for model
gender = st.selectbox("Select your gender", options=["Male", "Female"])
gender_male = 1 if gender == "Male" else 0  # Binary encoding for male
gender_female = 1 if gender == "Female" else 0  # Binary encoding for female

# Age input
age = st.number_input("Enter age", value=0)

# Family history of diabetes input (Yes/No) with binary encoding
family_diabetes = st.selectbox("Do you have a family history of diabetes?", options=["Yes", "No"])
family_diabetes_yes = 1 if family_diabetes == "Yes" else 0
family_diabetes_no = 1 if family_diabetes == "No" else 0

# Physical activity level selection (None, Less than half an hour, etc.), with corresponding binary encoding
physically_active = st.selectbox(
    "On average, how much are you physically active per day?", 
    options=["Less than half an hr", "None", "More than half an hr", "One hr or more"]
)
physically_active_less_than_half_hr = 1 if physically_active == "Less than half an hr" else 0
physically_active_none = 1 if physically_active == "None" else 0

# Blood pressure level input (High, Normal, Low), with corresponding binary encoding
bp_level = st.selectbox(
    label="What is your blood pressure level? (e.g., Normal, High, Low)",
    options=["High", "Normal", "Low"]
)
bp_level_high = 1 if bp_level == "High" else 0
bp_level_normal = 1 if bp_level == "Normal" else 0

# High blood pressure diagnosis (Yes/No) with binary encoding
high_bp = st.selectbox("Are you diagnosed with high blood pressure?", options=["Yes", "No"])
high_bp_yes = 1 if high_bp == "Yes" else 0
high_bp_no = 1 if high_bp == "No" else 0

# Collect user input for Continuous Glucose Monitoring (CGM) data. Expecting 60 values total (6 features for 10 timesteps)
time_series_input = st.text_area(
    "Enter your CGM (Continuous Glucose Monitoring) data: "
    "Please provide exactly 60 values, comma-separated, with 6 values per timestep for 10 timesteps."
    "Example: 100,105,110,120,115,100,..."
)

# When the user clicks the "Submit" button, proceed to handle the input data and make a prediction
if st.button("Submit"):
    if time_series_input:
        # Convert the comma-separated CGM data input into a numpy array
        time_series_values = np.array([float(x) for x in time_series_input.split(",")])

        # Ensure that the CGM data matches the expected length (60 values: 10 timesteps, 6 features per timestep)
        if len(time_series_values) < 60:
            st.error("Please provide exactly 60 values (10 timesteps, 6 values per timestep).")
        else:
            # Reshape the CGM data to match the model's input shape (1 sample, 10 timesteps, 6 features)
            time_series_values = time_series_values.reshape(1, 10, 6)

            # Prepare the fine-tuned features collected earlier, reshaped to match the model's input requirements
            fine_tuned_features = np.array([[pregnancies, bmi, sound_sleep, sleep,  
                                             gender_male, age, gender_female, family_diabetes_yes, 
                                             physically_active_less_than_half_hr, physically_active_none, 
                                             family_diabetes_no, bp_level_high, high_bp_no, high_bp_yes, 
                                             bp_level_normal]])

            # Scaling commented out for testing as it didn't work as expected
            # fine_tuned_features_scaled = scaler.transform(fine_tuned_features)
            # time_series_values_scaled = scaler.transform(time_series_values.reshape(-1, 6)).reshape(1, 10, 6)

            # Make a prediction using the model, providing both the time-series data and fine-tuned features
            prediction = model.predict([time_series_values, fine_tuned_features])

            # Get the index of the highest probability from the prediction (this corresponds to the predicted class)
            predicted_class = np.argmax(prediction)

            # Map the predicted class (0, 1, 2, or 3) to an actual diagnosis
            diagnosis_map = {
                0: "No diabetes",
                1: "Prediabetes",
                2: "Type 2 diabetes",
                3: "Gestational diabetes"
            }

            # Retrieve the corresponding diagnosis based on the predicted class
            diagnosis = diagnosis_map[predicted_class]

            # Display the predicted diagnosis on the UI
            st.write("Diagnosis:", diagnosis)

            # Store the input data and prediction results in MongoDB for future reference
            client = MongoClient(mongo_uri)  # Connect to MongoDB using the URI
            db = client['diabetes_data']  # Access the 'diabetes_data' database
            collection = db['predictions']  # Access the 'predictions' collection in the database

            # Insert the collected inputs and the prediction results into the MongoDB database
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
                "prediction": prediction.tolist(),  # Convert the prediction to a list before saving it
                "diagnosis": diagnosis  # Save the diagnosis result
            })
    else:
        st.write("Please provide CGM data.")  # Display error message if CGM data is missing