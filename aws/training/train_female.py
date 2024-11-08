import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# S3 bucket and path setup (SageMaker environment variables)
input_data_path = os.environ['INPUT_DATA_PATH']  # The path to the input data from the S3 bucket
output_data_path = os.environ['OUTPUT_DATA_PATH']  # The path for the trained model to be saved
model_s3_uri = os.environ['MODEL_S3_URI']  # The S3 URI for the previous model

columns_to_order = [
    'Age', 'HighBP', 'PhysicallyActive', 'BMI', 'Sleep', 'SoundSleep', 'JunkFood', 'BPLevel', 
    'Pregnancies', 'UriationFreq', 'HighChol', 'Fruits', 'Veggies', 'GenHlth', 'PhysHlth', 
    'Gestation in previous Pregnancy', 'PCOS', 'sudden weight loss', 'visual blurring', 
    'delayed healing', 'Pregnant'
]

# Step 1: Load the data from S3 (data is expected in CSV format)
print("Loading data...")
data = pd.read_csv(input_data_path)

data = data[columns_to_order]

# Step 2: Preprocess the data (you might need to adjust depending on your dataset)
# Assuming the dataset includes a target column 'prediction' (your output variable)
X = data.drop(columns=['prediction'], axis=1)
y = data['prediction']

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define and load the existing XGBoost model
print("Loading the existing model...")
model = xgb.XGBClassifier(eval_metric="mlogloss", objective="multi:softmax", num_class=4)
model.load_model(model_s3_uri)  # Load the previous model

# Step 4: Retrain the model
print("Retraining the model with new data...")
model.fit(X_train, y_train)

# Step 5: Validate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Step 6: Save the model to the specified S3 location
print("Saving model to S3...")
model.save_model(os.path.join(output_data_path, 'xgboost_female.json'))

# Step 7: Return the location of the saved model
print(f"Model saved at: {output_data_path}/xgboost_female.json")
