import streamlit as st  # Streamlit is used for building interactive web applications in Python
import numpy as np  # NumPy is used for numerical operations, mainly handling arrays
from pymongo import MongoClient  # MongoDB client for connecting to the MongoDB database
from keras.models import load_model  # type: ignore # Used to load a pre-trained Keras deep learning model
from sklearn.preprocessing import StandardScaler  # StandardScaler for feature scaling
import joblib  # Used for loading pre-trained scalers and preprocessor objects
import pandas as pd  # Pandas is used for handling data in tabular format
from dotenv import load_dotenv  # Load environment variables from a .env file for secure configuration
import os  # For interacting with environment variables

# Load environment variables from the .env file (e.g., database connection strings)
load_dotenv()

# Get MongoDB URI from environment variables for a secure connection to the MongoDB database
mongo_uri = os.getenv('MONGO_DB_CONN_URL')

# Load the pre-trained Keras deep learning model (assumed to be a .h5 model file)
model = load_model('final_deep_learning_model.h5')

# Load the pre-trained scalers and top feature list using joblib
structured_data_preprocessor = joblib.load('structured_data_preprocessor.pkl')  # Preprocessor for structured input
time_series_scaler = joblib.load('time_series_scaler.pkl')  # Scaler for time-series input (CGM data)
top_features_list = joblib.load('top_features_list.pkl')  # List of top features for the model

# Collect user inputs through Streamlit's interactive interface
pregnancies = st.number_input("How many pregnancies have you had in total?", min_value=0, value=0)

# Collect height and weight to calculate BMI
height_in = st.number_input("Enter your height (in inches):", min_value=0.0, value=0.0)
weight_lb = st.number_input("Enter your weight (in pounds):", min_value=0.0, value=0.0)

# Calculate BMI if both height and weight are provided
if height_in > 0 and weight_lb > 0:
    # BMI calculation using the imperial system: (weight in pounds * 703) / (height in inches)^2
    bmi = (weight_lb * 703) / (height_in ** 2)
    st.write(f"Your calculated BMI is: {bmi:.2f}")
else:
    st.write("Please enter valid height and weight to calculate BMI.")

# Collect inputs related to sleep quality and quantity
sound_sleep = st.number_input("How many hours of sound sleep do you get on average?", value=0.0)
sleep = st.number_input("Enter your total sleep time on average per day (in hours)", value=0.0)

# Gender selection input (binary choice: Male/Female)
gender = st.selectbox("Select your gender", options=["Male", "Female"])

# Age input
age = st.number_input("Enter your age", value=0)

# Family history of diabetes input (binary choice: Yes/No)
family_diabetes = st.selectbox("Do you have a family history of diabetes?", options=["yes", "no"])

# Physical activity level selection with multiple options
physically_active = st.selectbox(
    "On average, how much physical activity do you get per day?",
    options=["less than half an hr", "none", "more than half an hr", "one hr or more"]
)

# Blood pressure level selection (e.g., High, Normal, Low)
bp_level = st.selectbox(
    label="What is your blood pressure level?",
    options=["high", "normal", "low"]
)

# Diagnosis of high blood pressure (binary choice: Yes/No)
high_bp = st.selectbox("Have you been diagnosed with high blood pressure?", options=["yes", "no"])

# Collect user input for Continuous Glucose Monitoring (CGM) data
# Expecting 60 values: 6 features for 10 timesteps (comma-separated)
time_series_input = st.text_area(
    "Enter your CGM (Continuous Glucose Monitoring) data: "
    "Please provide exactly 60 values, comma-separated, with 6 values per timestep for 10 timesteps."
    "Example: 100,105,110,120,115,100,..."
)

# Default values for 'Smoking' and 'Alcohol' since these are not provided by the user
smoking = "no"  # Default value for smoking
alcohol = "no"  # Default value for alcohol

# Organize user inputs into a dictionary matching the expected feature set
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

# Convert the input data into a Pandas DataFrame
input_data = pd.DataFrame([input_data_dict])

# Apply the preprocessor to the structured data (scales numerical features, encodes categorical features)
structured_input_transformed = structured_data_preprocessor.transform(input_data)

# Convert transformed input into a DataFrame for easier manipulation
input_data_transformed_df = pd.DataFrame(structured_input_transformed, columns=structured_data_preprocessor.get_feature_names_out())

# Select only the columns (features) that are in the top features list for prediction
filtered_input_data = input_data_transformed_df[top_features_list]

# Convert the filtered DataFrame to a NumPy array for model input
filtered_input_data_array = filtered_input_data.to_numpy()

# When the user clicks the "Submit" button, process the inputs and make a prediction
if st.button("Submit"):
    if time_series_input:
        # Convert the CGM data input into a NumPy array
        time_series_values = np.array([float(x) for x in time_series_input.split(",")])

        # Ensure the input CGM data contains exactly 60 values (6 values per timestep for 10 timesteps)
        if len(time_series_values) < 60:
            st.error("Please provide exactly 60 values (10 timesteps, 6 values per timestep).")
        else:
            # Reshape the CGM data to match the model's expected input shape (1 sample, 10 timesteps, 6 features)
            time_series_values = time_series_values.reshape(1, 10, 6)

            # Scale the time-series data
            time_series_scaled = time_series_scaler.transform(time_series_values.reshape(-1, 6)).reshape(1, 10, 6)

            # Make a prediction using the model (both time-series and structured inputs)
            prediction = model.predict([time_series_scaled, filtered_input_data_array])

            # Get the predicted class index (e.g., 0, 1, 2, 3)
            predicted_class = np.argmax(prediction)

            # Map the predicted class index to an actual diagnosis
            diagnosis_map = {
                0: "No diabetes",
                1: "Prediabetes",
                2: "Type 2 diabetes",
                3: "Gestational diabetes"
            }

            # Retrieve the corresponding diagnosis
            diagnosis = diagnosis_map[predicted_class]

            # Display the predicted diagnosis on the interface
            st.write("Diagnosis:", diagnosis)

            # Store the input data and prediction results in MongoDB
            client = MongoClient(mongo_uri)  # Connect to MongoDB
            db = client['diabetes_data']  # Access the 'diabetes_data' database
            collection = db['predictions']  # Access the 'predictions' collection

            input_data_dict['cgm_data'] = time_series_input  # Include CGM data in the stored input
            # Insert the input data and the prediction result into the MongoDB database
            collection.insert_one({
                "input_values": input_data_dict,
                "prediction": prediction.tolist(),  # Convert prediction to a list for storage
                "diagnosis": diagnosis  # Save the diagnosis result
            })
    else:
        st.write("Please provide CGM data.")  # Error message if CGM data is missing
