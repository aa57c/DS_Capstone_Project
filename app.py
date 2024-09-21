import streamlit as st  # Streamlit is used for building web applications easily with Python
import numpy as np  # Numpy is used for numerical operations, mainly handling arrays in this script
from pymongo import MongoClient  # MongoDB client for connecting and interacting with the database
from keras.models import load_model  # type: ignore # For loading the pre-trained Keras deep learning model
from sklearn.preprocessing import StandardScaler  # StandardScaler for feature scaling during model input
import joblib
import pandas as pd

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
structured_data_preprocessor = joblib.load('structured_data_preprocessor.pkl')
time_series_scaler = joblib.load('time_series_scaler.pkl')
top_features_list = joblib.load('top_features_list.pkl')


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

# Age input
age = st.number_input("Enter age", value=0)

# Family history of diabetes input (Yes/No) with binary encoding
family_diabetes = st.selectbox("Do you have a family history of diabetes?", options=["yes", "no"])


# Physical activity level selection (None, Less than half an hour, etc.), with corresponding binary encoding
physically_active = st.selectbox(
    "On average, how much are you physically active per day?", 
    options=["less than half an hr", "none", "more than half an hr", "one hr or more"]
)

# Blood pressure level input (High, Normal, Low), with corresponding binary encoding
bp_level = st.selectbox(
    label="What is your blood pressure level? (e.g., Normal, High, Low)",
    options=["high", "normal", "low"]
)

# High blood pressure diagnosis (Yes/No) with binary encoding
high_bp = st.selectbox("Are you diagnosed with high blood pressure?", options=["yes", "no"])

# Collect user input for Continuous Glucose Monitoring (CGM) data. Expecting 60 values total (6 features for 10 timesteps)
time_series_input = st.text_area(
    "Enter your CGM (Continuous Glucose Monitoring) data: "
    "Please provide exactly 60 values, comma-separated, with 6 values per timestep for 10 timesteps."
    "Example: 100,105,110,120,115,100,..."
)

# Assign default values for 'Smoking' and 'Alcohol' since the user doesn't provide these inputs
smoking = "no"  # Assuming 'No' as default
alcohol = "no"  # Assuming 'No' as default

# Organize user inputs into a dictionary matching the expected features
input_data_dict = {
    'Pregancies': pregnancies,
    'BMI': bmi,
    'SoundSleep': sound_sleep,
    'Sleep': sleep,
    'Gender': gender,
    'Age': age,
    'Family_Diabetes': family_diabetes,
    'PhysicallyActive': physically_active,
    'BPLevel': bp_level,
    'highBP': high_bp,
    'Smoking': smoking,   # Default value
    'Alcohol': alcohol    # Default value
}

# Convert the input data into a DataFrame
input_data = pd.DataFrame([input_data_dict])

# Apply the preprocessor to transform the data
# This will scale the numerical features and one-hot encode the categorical ones
structured_input_transformed = structured_data_preprocessor.transform(input_data)

# After transforming the input data, filter the top features for prediction
input_data_transformed_df = pd.DataFrame(structured_input_transformed, columns=structured_data_preprocessor.get_feature_names_out())

# Select only the columns (features) that are in the top features list
filtered_input_data = input_data_transformed_df[top_features_list]

# Convert filtered data back to a numpy array for model input
filtered_input_data_array = filtered_input_data.to_numpy()

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

            # Scale the time-series data
            time_series_scaled = time_series_scaler.transform(time_series_values.reshape(-1, 6)).reshape(1, 10, 6)

            # Make a prediction using the model, providing both the time-series data and fine-tuned features
            prediction = model.predict([time_series_scaled, filtered_input_data_array])

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

            input_data_dict['cgm_data'] = time_series_input
            # Insert the collected inputs and the prediction results into the MongoDB database
            collection.insert_one({
                "input_values": input_data_dict,
                "prediction": prediction.tolist(),  # Convert the prediction to a list before saving it
                "diagnosis": diagnosis  # Save the diagnosis result
            })
    else:
        st.write("Please provide CGM data.")  # Display error message if CGM data is missing