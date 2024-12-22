### AI Powered Diabetes Prediction and Health Coach App

## Demo: https://youtu.be/ilnozF_kpas

## Overview

This repository contains a Streamlit application designed for predicting diabetes using pretrained XGBoost and LSTM models. The app collects user input, processes the data, and makes combined predictions from both models. User inputs and predictions are also stored in a MongoDB database for future retraining and analysis.

## Features
- User-friendly interface for data entry and prediction.
- Combines predictions from XGBoost and LSTM models.
- Stores user data and prediction results in MongoDB.
- Supports model retraining using recent user data stored in MongoDB and S3.

## Prerequisites
- Python 3.8 or higher
- MongoDB (for data storage)
- Ollama server (required for Large Language Model integrations)

## Installation
1. Clone the Repository

       git clone [https://github.com/yourusername/diabetes-prediction-app.git] (https://github.com/aa57c/DS_Capstone_Project.git)
       cd DS_Capstone_Project

3. Set up .ENV file for the database connection url
   shell
   
       touch .env
   
   .env
   
       MONGO_DB_CONN_URL=mongodb+srv://<username>:<password>@cluster0.yhab1.mongodb.net/


5. Install Dependencies
   
   shell

       touch requirements.txt

   requirements.txt
       
       streamlit==1.23.1
       numpy==1.24.2
       pandas==2.1.1
       requests==2.31.0
       pymongo==4.4.0
       python-dotenv==1.0.0
       joblib==1.2.0
       xgboost==1.7.6
       tensorflow==2.13.0
       scikit-learn==1.2.2
   
    shell
       
       pip install -r requirements.txt

7. Download and Install the Ollama Server
   
   To enhance your application with the LLM, you will need the Ollama server. Follow the link below for installation instructions:
   https://ollama.com/

   Make sure the server is running before you run the application.
   
9. Run the application

       streamlit run app.py

## Configuration

- Ensure that MongoDB is running locally or update the connection URI in the application code for remote access.
- Modify the configuration settings in app.py as needed for your environment (e.g., model file paths and database credentials).

## Usage

1. Open your web browser and navigate to http://localhost:8501.
2. Fill out the input fields and submit the form to receive a prediction.
3. View the results and prediction confidence, which will be logged in MongoDB.
    

