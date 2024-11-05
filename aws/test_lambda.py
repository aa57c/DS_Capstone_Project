import boto3
import os
import pandas as pd
from pymongo import MongoClient
from io import StringIO
import datetime

# Initialize S3 client
s3 = boto3.client('s3')

def save_to_s3(dataframe, bucket, key):
    """Utility function to save a DataFrame to S3 as CSV."""
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f"Data uploaded to {key}")

def lambda_handler():

    # Fetch data from MongoDB
    try:
        # MongoDB connection (MongoDB URI should be stored as an environment variable)
        mongo_uri = 'mongodb+srv://aa57c:DXGymy4BJ97XAYYo@cluster0.yhab1.mongodb.net/'
        client = MongoClient(mongo_uri)
        db = client['DiabetesRepo']
        collection = db['Diabetes_Prediction_Data']
    
        # Define S3 bucket
        s3_bucket = 'diabetes-prediction-data'  # Bucket name
        # Retrieve all documents
        data = list(collection.find())

        if not data:
            print("No data found in MongoDB collection.")
            return {'statusCode': 200, 'body': "No data found in MongoDB collection."}
        
        # Initialize lists to store structured data and CGM data separately
        male_data_entries = []
        female_data_entries = []
        cgm_data_entries = []

        # Iterate through each user document
        for document in data:
            username = document.get("username", "unknown_user")
            
            # Iterate over each data entry within a user's document
            for entry in document.get("data", []):
                # Split data based on gender
                gender = entry.get("gender", "").lower()
                
                # Process structured data
                structured_data = {k: v for k, v in entry.items() if k not in ["cgm", "class_probabilities", "diagnosis", "recommendations", "_id", "timestamp", "gender"]}
                
                if gender == "male":
                    male_data_entries.append(structured_data)
                elif gender == "female":
                    female_data_entries.append(structured_data)

                # Process CGM data (if available)
                if "cgm" in entry:
                    cgm_data = entry["cgm"]
                    # Create a DataFrame from CGM data, where each CGM value gets a separate column
                    cgm_columns = {f"cgm_{i+1}": value for i, value in enumerate(cgm_data)}
                    cgm_entry = {"username": username}
                    cgm_entry.update(cgm_columns)
                    cgm_data_entries.append(cgm_entry)


        # Convert lists to DataFrames and upload to S3 as separate files
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        if male_data_entries:
            df_male = pd.DataFrame(male_data_entries)
            save_to_s3(df_male, s3_bucket, f'male-data/male_data_{current_date}.csv')

        if female_data_entries:
            df_female = pd.DataFrame(female_data_entries)
            save_to_s3(df_female, s3_bucket, f'female-data/female_data_{current_date}.csv')

        if cgm_data_entries:
            df_cgm = pd.DataFrame(cgm_data_entries)
            save_to_s3(df_cgm, s3_bucket, f'cgm-data/cgm_data_{current_date}.csv')

        return {'statusCode': 200, 'body': "Data successfully processed and uploaded to S3"}
    
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

lambda_handler()

