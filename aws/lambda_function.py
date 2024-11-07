import boto3
import pandas as pd
from pymongo import MongoClient
from io import StringIO
import datetime
import os

# Initialize S3 client
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker', region_name='us-east-1')

def delete_old_files(bucket, prefix):
    """Delete old CSV files in a specific S3 bucket folder."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            s3.delete_object(Bucket=bucket, Key=obj['Key'])
            print(f"Deleted old file: {obj['Key']}")

def save_to_s3(dataframe, bucket, key):
    """Utility function to save a DataFrame to S3 as CSV."""
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f"Data uploaded to {key}")

def start_sagemaker_training(job_name, image_uri, input_data_s3_uri, output_data_s3_uri, model_s3_uri, role_arn, script_uri):
    response = sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            'TrainingImage': image_uri,  # The XGBoost image URI
            'TrainingInputMode': 'File',
            'ScriptModeConfig': {
                'EntryPoint': script_uri,  # Path to your custom script
            }
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_data_s3_uri,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv'
            }
        ],
        OutputDataConfig={
            'S3OutputPath': output_data_s3_uri
        },
        ResourceConfig={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 10800
        },
        HyperParameters={
            'model_s3_uri': model_s3_uri,  # Pass model URI for retraining
        }
    )
    print(f"Started SageMaker training job: {job_name}")
    return response


def lambda_handler(event, _):
    MONGO_URI = os.environ.get("MONGO_CONN_URL")
    S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
    SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
    
    if not MONGO_URI or not S3_BUCKET or not SAGEMAKER_ROLE_ARN:
        print("One or more environment variables are missing.")
        return {'statusCode': 500, 'body': "Missing required environment variables."}
    
    print(f"MONGO_URI: {MONGO_URI}")
    print(f"S3_BUCKET_NAME: {S3_BUCKET}")
    print(f"SAGEMAKER_ROLE_ARN: {SAGEMAKER_ROLE_ARN}")

    # Delete old CSVs before uploading new data
    delete_old_files(S3_BUCKET, 'male-data/')
    delete_old_files(S3_BUCKET, 'female-data/')
    delete_old_files(S3_BUCKET, 'cgm-data/')

    # Fetch data from MongoDB
    try:
        print("Starting Lambda function.")
        
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient(MONGO_URI)
        db = client['DiabetesRepo']
        collection = db['Diabetes_Prediction_Data']
        
        # Retrieve all documents
        print("Retrieving data from MongoDB collection...")
        data = list(collection.find())

        if not data:
            print("No data found in MongoDB collection.")
            return {'statusCode': 200, 'body': "No data found in MongoDB collection."}

        # Initialize dictionaries to store latest data by username
        male_data_entries = {}
        female_data_entries = {}
        cgm_data_entries = {}

        # Keys to exclude from the final upload
        exclude_keys = {"cgm", "diagnosis", "class_probabilities", "timestamp", "gender", "recommendations", "_id"}

        # Iterate through each user document
        print("Processing documents...")
        for document in data:
            username = document.get("username", "unknown_user")
            
            # Iterate over each data entry within a user's document
            for entry in document.get("data", []):
                timestamp = entry.get("timestamp", None)
                gender = entry.get("gender", "").lower()

                if timestamp is None:
                    continue  # Skip entries with no timestamp

                # Convert timestamp to a datetime object if it’s a string
                if isinstance(timestamp, str):
                    entry["timestamp"] = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
                else:
                    entry["timestamp"] = timestamp  # Already a datetime object

                # Keep the latest entry by timestamp for each user and gender
                if gender == "male":
                    if username not in male_data_entries or male_data_entries[username]["timestamp"] < entry["timestamp"]:
                        male_data_entries[username] = entry
                elif gender == "female":
                    if username not in female_data_entries or female_data_entries[username]["timestamp"] < entry["timestamp"]:
                        female_data_entries[username] = entry

                # Process CGM data (if available)
                if "cgm" in entry:
                    cgm_data = entry["cgm"]
                    cgm_entry = {"username": username}  # Removed timestamp here
                    cgm_columns = {f"cgm_{i+1}": value for i, value in enumerate(cgm_data)}
                    cgm_entry.update(cgm_columns)

                    if username not in cgm_data_entries or cgm_data_entries[username]["timestamp"] < entry["timestamp"]:
                        cgm_data_entries[username] = cgm_entry

        print("Data processing completed.")
        
        # Prepare final lists of data without excluded keys
        male_data = [{k: v for k, v in entry.items() if k not in exclude_keys} for entry in male_data_entries.values()]
        female_data = [{k: v for k, v in entry.items() if k not in exclude_keys} for entry in female_data_entries.values()]
        cgm_data = [v for v in cgm_data_entries.values()]

        # Convert to DataFrames and upload to S3 as separate files
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        if male_data:
            print("Uploading male data to S3...")
            df_male = pd.DataFrame(male_data)
            save_to_s3(df_male, S3_BUCKET, f'male-data/male_data_{current_date}.csv')

        if female_data:
            print("Uploading female data to S3...")
            df_female = pd.DataFrame(female_data)
            save_to_s3(df_female, S3_BUCKET, f'female-data/female_data_{current_date}.csv')

        if cgm_data:
            print("Uploading CGM data to S3...")
            df_cgm = pd.DataFrame(cgm_data)
            df_cgm.drop(columns=['timestamp'], errors='ignore', inplace=True)  # Drop timestamp column if exists
            save_to_s3(df_cgm, S3_BUCKET, f'cgm-data/cgm_data_{current_date}.csv')

        print("Data successfully processed and uploaded to S3.")
        
        xgboost_image_uri = sagemaker.image_uris.retrieve(
            framework='xgboost',
            region='us-east-1',
            version='1.5-1'
        )

        # Define the location of your custom training script on S3
        xgboost_script_female_uri = 's3://path/to/your/custom/train.py'
        # xgboost_script_male_uri = 's3://path/to/your/custom/train.py'
        
        # Trigger SageMaker job for retraining ONLY XGBoost models
        job_name = f"retrain-xgboost-female-model-{current_date}"
        image_uri = xgboost_image_uri  # Update with the XGBoost training image URI
        input_data_s3_uri = f"s3://{S3_BUCKET}/female-data/"
        output_data_s3_uri = f"s3://{S3_BUCKET}/new-models/female/"
        model_s3_uri = f"s3://{S3_BUCKET}/old-models/female/"  # Point to XGBoost models only
        script_uri = xgboost_script_female_uri  # Path to your script in S3
        
        start_sagemaker_training(job_name, image_uri, input_data_s3_uri, output_data_s3_uri, model_s3_uri, SAGEMAKER_ROLE_ARN, script_uri)
        
        print("Successfully started Sagemaker jobs.")
        
        return {'statusCode': 200, 'body': "Data successfully processed, uploaded to S3, and starting Sagemaker jobs"}

    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
