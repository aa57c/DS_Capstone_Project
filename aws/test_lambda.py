import boto3
import pandas as pd
from pymongo import MongoClient
from io import StringIO
import datetime
from dotenv import load_dotenv
import os

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
        # Load environment variables
        load_dotenv()
        MONGO_URI = os.getenv("MONGO_DB_CONN_URL")
        client = MongoClient(MONGO_URI)
        db = client['DiabetesRepo']
        collection = db['Diabetes_Prediction_Data']

        # Define S3 bucket
        s3_bucket = 'diabetes-prediction-data'  # Bucket name
        # Retrieve all documents
        data = list(collection.find())

        if not data:
            print("No data found in MongoDB collection.")
            return {'statusCode': 200, 'body': "No data found in MongoDB collection."}

        # Initialize dictionaries to store latest data by username
        male_data_entries = {}
        female_data_entries = {}
        cgm_data_entries = {}

        # Iterate through each user document
        for document in data:
            username = document.get("username", "unknown_user")
            
            # Iterate over each data entry within a user's document
            for entry in document.get("data", []):
                timestamp = entry.get("timestamp", None)
                gender = entry.get("gender", "").lower()

                if timestamp is None:
                    continue  # Skip entries with no timestamp

                # Find the latest entry for each user by timestamp
                entry["timestamp"] = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')  # Ensure timestamp is a datetime object

                # Compare and store the latest entry per user
                if gender == "male":
                    if username not in male_data_entries or male_data_entries[username]["timestamp"] < entry["timestamp"]:
                        male_data_entries[username] = entry
                elif gender == "female":
                    if username not in female_data_entries or female_data_entries[username]["timestamp"] < entry["timestamp"]:
                        female_data_entries[username] = entry

                # Process CGM data (if available)
                if "cgm" in entry:
                    cgm_data = entry["cgm"]
                    cgm_entry = {"username": username, "timestamp": entry["timestamp"]}
                    cgm_columns = {f"cgm_{i+1}": value for i, value in enumerate(cgm_data)}
                    cgm_entry.update(cgm_columns)

                    if username not in cgm_data_entries or cgm_data_entries[username]["timestamp"] < entry["timestamp"]:
                        cgm_data_entries[username] = cgm_entry

        # Convert to DataFrames and upload to S3 as separate files
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        # Prepare data for male and female
        male_data = [v for v in male_data_entries.values()]
        female_data = [v for v in female_data_entries.values()]
        cgm_data = [v for v in cgm_data_entries.values()]

        if male_data:
            df_male = pd.DataFrame(male_data)
            save_to_s3(df_male, s3_bucket, f'male-data/male_data_{current_date}.csv')

        if female_data:
            df_female = pd.DataFrame(female_data)
            save_to_s3(df_female, s3_bucket, f'female-data/female_data_{current_date}.csv')

        if cgm_data:
            df_cgm = pd.DataFrame(cgm_data)
            save_to_s3(df_cgm, s3_bucket, f'cgm-data/cgm_data_{current_date}.csv')

        return {'statusCode': 200, 'body': "Data successfully processed and uploaded to S3"}
    
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

lambda_handler()
