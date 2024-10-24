import pandas as pd
import numpy as np

# Load the imputed female data with Patient_ID
df1 = pd.read_csv('imputed_female_data_updated.csv')

'''
# Simulation parameters
num_readings_per_day = 288  # Simulated 24 hours of CGM readings (every 5 minutes)
num_days = 7  # Simulate one week of data
total_readings = num_readings_per_day * num_days  # Total number of CGM readings

# Function to simulate glucose levels
def simulate_glucose(level_mean, level_std, num_patients, total_readings):
    return np.random.normal(loc=level_mean, scale=level_std, size=(num_patients, total_readings))

# Simulate Meal Effects
def simulate_meal_effects(time_series, meal_times, spike_magnitude, std):
    for meal_time in meal_times:
        spike_duration = 10  # Simulating 10 minutes of elevated glucose after meal
        start_spike = max(0, meal_time - spike_duration // 2)
        end_spike = min(len(time_series), meal_time + spike_duration // 2)
        time_series[start_spike:end_spike] += np.random.normal(loc=spike_magnitude, scale=std, size=(end_spike - start_spike))
    return time_series

# Simulate Circadian Rhythms
def circadian_rhythm(time_series, night_start=220, night_end=60, night_dip=15, morning_surge=20):
    time_series[night_start:] -= night_dip  # Nighttime dip
    time_series[:night_end] -= night_dip
    time_series[night_end:night_end + 20] += morning_surge  # Morning surge
    return time_series

# Simulate Physical Activity
def simulate_physical_activity(time_series, activity_times, drop_magnitude):
    for activity_time in activity_times:
        time_series[activity_time:activity_time + 15] -= drop_magnitude  # Drop after physical activity
    return time_series

# Simulate Stress Events
def simulate_stress_events(time_series, stress_times, spike_magnitude):
    for stress_time in stress_times:
        time_series[stress_time:stress_time + 10] += spike_magnitude  # Glucose spike due to stress
    return time_series

# Add Individual Variability
def individual_variability(glucose_data, variability_factor=0.1):
    variability = np.random.normal(1, variability_factor, size=glucose_data.shape)
    return glucose_data * variability

# Add Sensor Noise
def add_sensor_noise(glucose_data, noise_level=5):
    noise = np.random.normal(0, noise_level, glucose_data.shape)
    return glucose_data + noise

# Simulate Diabetes-Specific Patterns
def diabetes_specific_patterns(time_series, diabetes_type):
    meal_times = [60, 120, 180]  # Meal times (breakfast, lunch, dinner)
    
    if diabetes_type == 0:  # No diabetes
        time_series = simulate_meal_effects(time_series, meal_times, spike_magnitude=30, std=5)
        time_series = circadian_rhythm(time_series, night_dip=10, morning_surge=15)
        
    elif diabetes_type == 1:  # Prediabetes
        time_series = simulate_meal_effects(time_series, meal_times, spike_magnitude=50, std=10)
        time_series = circadian_rhythm(time_series, night_dip=12, morning_surge=18)
        
    elif diabetes_type == 2:  # Type 2 Diabetes
        time_series = simulate_meal_effects(time_series, meal_times, spike_magnitude=70, std=15)
        time_series = circadian_rhythm(time_series, night_dip=5, morning_surge=30)
        time_series = simulate_stress_events(time_series, stress_times=[300, 450], spike_magnitude=30)
        
    elif diabetes_type == 3:  # Gestational Diabetes
        time_series = simulate_meal_effects(time_series, meal_times, spike_magnitude=60, std=12)
        time_series = circadian_rhythm(time_series, night_dip=10, morning_surge=25)

    # Simulate random physical activity
    time_series = simulate_physical_activity(time_series, activity_times=[100, 250], drop_magnitude=20)
    
    return time_series

# Simulate CGM data for each patient in df1
patient_glucose_data = []

# Loop through each patient and simulate their glucose readings
for _, patient in df1.iterrows():
    print(f"Simulating glucose data for Patient_ID: {patient['Patient_ID']}")

    diabetes_status = patient['Diabetes_Status']
    
    # Simulate glucose levels
    glucose_levels = simulate_glucose(level_mean=100, level_std=15, num_patients=1, total_readings=total_readings)[0]
    
    # Apply diabetes-specific patterns
    glucose_levels = diabetes_specific_patterns(glucose_levels, diabetes_status)
    glucose_levels = add_sensor_noise(glucose_levels)  # Add sensor noise
    glucose_levels = individual_variability(glucose_levels, variability_factor=0.2)  # Add individual variability
    
    # Store the glucose data with Patient_ID
    patient_glucose_data.append({
        'Patient_ID': patient['Patient_ID'],
        'Glucose_Readings': glucose_levels
    })
    print(f"Completed Glucose Readings for Patient_ID: {patient['Patient_ID']}")

# Convert glucose data to DataFrame
glucose_df = pd.DataFrame(patient_glucose_data)

# Convert the glucose readings array into separate columns
glucose_readings_df = pd.DataFrame(glucose_df['Glucose_Readings'].to_list())
glucose_readings_df.columns = [f'Reading_{i+1}' for i in range(glucose_readings_df.shape[1])]

# Combine Patient_ID and simulated glucose readings
final_df = pd.concat([glucose_df['Patient_ID'], glucose_readings_df], axis=1)

# Save the final DataFrame to a new CSV file
final_df.to_csv('female_patients_with_simulated_glucose_data.csv', index=False)

print("Synthetic CGM data simulation complete and saved to 'patients_with_simulated_glucose_data.csv'.")
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the newly created CSV
df = pd.read_csv('female_patients_with_simulated_glucose_data.csv')

# Assume you have 'Diabetes_Status' in your original df1, so you might need to merge it back
# Merge or map diabetes status back if it's not included already
diabetes_status = df1[['Patient_ID', 'Diabetes_Status']]  # Assuming df1 contains original 'Diabetes_Status'
df = pd.merge(df, diabetes_status, on='Patient_ID')

# Separate features and labels
X = df.drop(columns=['Patient_ID', 'Diabetes_Status'])  # Glucose readings as features
y = df['Diabetes_Status']  # Target labels

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features (optional but often useful for deep learning)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape X into the format needed for LSTM (samples, timesteps, features)
timesteps = X_train.shape[1]  # Number of time steps equals number of glucose readings
X_train = X_train.reshape((X_train.shape[0], timesteps, 1))
X_test = X_test.reshape((X_test.shape[0], timesteps, 1))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Build the LSTM model
model = Sequential()

# Add LSTM layers
model.add(LSTM(128, return_sequences=True, input_shape=(timesteps, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output layer for classification
model.add(Dense(4, activation='softmax'))  # Assuming 4 output classes for diabetes types (0, 1, 2, 3)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=128, 
                    callbacks=[reduce_lr, early_stopping])

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# You can further evaluate the performance using classification metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_labels))






