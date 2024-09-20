# # # Diabetes Prediction Using Time Series and Fine-Tuned Features

# # Project Overview
This project implements a predictive model to determine whether an individual is likely to have prediabetes, type 2 diabetes, or gestational diabetes based on a combination of structured input features and time-series data from continuous glucose monitoring (CGM). The application uses a deep learning model to process the inputs and provide a diagnosis. It also stores the input data and predictions in a MongoDB database.

# # Project Structure
# 1. Frontend (Streamlit)
The frontend of this application is developed using Streamlit, which provides a simple web interface for users to input their data. Users are prompted to provide structured data such as:

1. Number of pregnancies
2. BMI (Body Mass Index)
3. Sleep duration
4. Blood pressure levels
5. Gender
6. Age
7. Physical activity levels

Additionally, users input time-series data from CGM (Continuous Glucose Monitoring), consisting of multiple glucose readings over time.

Once the user inputs all the necessary data, the model processes it and provides a prediction, displaying whether the user has no diabetes, prediabetes, type 2 diabetes, or gestational diabetes.

# 2. Model (Deep Learning Model)
The deep learning model was trained using Keras/TensorFlow and is saved in the final_deep_learning_model.h5 file. This model takes in two sets of input:

  Structured data (fine-tuned features like age, BMI, sleep patterns, etc.)
  Time-series CGM data (10 timesteps, 6 values per timestep)

The model is designed to handle both types of inputs simultaneously and generate a prediction based on the user data.

# 3. Database (MongoDB)
The application uses MongoDB to store user input data and model predictions. When a prediction is made, the input values (including CGM data) and the corresponding diagnosis are saved to a MongoDB collection. With MongoDB and Streamlit, there is no need for an API layer to connect the two. The frontend changes reflect in the backend directly.

# 4. Connecting Frontend, Model, and Database

The Streamlit frontend interacts with the deep learning model and MongoDB database in the following flow:

1. The user submits input data via the Streamlit web interface.
2. The data is processed and passed to the deep learning model (loaded from final_deep_learning_model.h5).
3. The model predicts the likelihood of diabetes (or no diabetes).
4. The results, along with the user inputs, are saved in the MongoDB database for record-keeping and future analysis.

# # Running the Application

# Prerequisites
1. Python 3.x
2. Libraries:
   a. Streamlit (pip install streamlit)
   b. NumPy (pip install numpy)
   c. TensorFlow/Keras (pip install tensorflow)
   d. PyMongo (pip install pymongo)
   e. MongoDB (locally or remotely hosted)

# Installation
1. Clone the repository:
   git clone <repository_url>
2. Navigate to the project directory:
   cd CS_5588_DS_Capstone_Assignments
3. Install required dependencies
4. Set up your MongoDB instance and configure the connection string in app.py where the MongoClient is initialized:
   client = MongoClient("<your_mongo_db_connection_string>")

# Running the Application
To run the application, execute the following command:
streamlit run app.py
This command will launch the Streamlit web application. You can access the app in your browser, where you can input the data and receive a diagnosis.

# Model Deployment
The deep learning model (final_deep_learning_model.h5) is loaded in app.py using Keras’ load_model function. This allows the model to make predictions whenever the user submits new data through the Streamlit interface. The model handles both structured input features and time-series data simultaneously.

# Accessing MongoDB
The application stores the input data and predictions in a MongoDB database. Ensure your MongoDB instance is running and configured properly in app.py. You can view saved records by connecting to your MongoDB instance and checking the specified collection.

# Prototype Description and Project Goals
This triangle model prototype connects the frontend (Streamlit), the deep learning model (TensorFlow/Keras), and the database (MongoDB) to form a robust system for diabetes prediction. By combining the following components:

  Frontend: Allows users to easily input data and receive real-time predictions.
  Model: Provides accurate predictions based on fine-tuned input features and CGM data.
  Database: Ensures all inputs and results are stored securely for future analysis.

The triangle model supports the project goals by demonstrating how machine learning models can be used to make real-time predictions based on both structured and time-series data. This prototype showcases the potential to integrate AI models into healthcare applications, providing a foundation for more comprehensive predictive systems in the future.




