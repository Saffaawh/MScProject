import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# --- 1. Load Historical Data ---
try:
    df_history = pd.read_csv('historical_mri_appointment_dataset_raw.csv')
    print("Historical Dataset Loaded.")
    print(f"Total Patients Available: {len(df_history)}")
    
    # Select relevant columns
    # Ensure 'Delay Reason' is consistent (handle NaNs or whitespace)
    if 'DelayReason' in df_history.columns:
        df_history.rename(columns={'DelayReason': 'Delay Reason'}, inplace=True)
    
    df_history['Delay Reason'] = df_history['Delay Reason'].fillna('None').astype(str).str.strip()
    
    # We use the WHOLE historical dataset as the base
    df_patient_details = df_history[['PatientNumber', 'Age', 'Gender', 'Delay Reason']]

except FileNotFoundError:
    print("Error: 'historical_mri_appointment_dataset_raw.csv' not found.")
    exit()

# --- 2. Define Clinical Logic ---
# Mapping: Indication -> {Scan Type: Probability}
clinical_mapping = { 
    'headache': {'Brain': 0.8, 'Brain with Contrast': 0.2},
    'neck pain': {'Cervical Spine': 0.8, 'Cervical Spine with Contrast': 0.2},
    'tumor': {'Brain with Contrast': 0.4, 'Cervical Spine with Contrast': 0.1, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine with Contrast': 0.1, 'Whole Spine with Contrast': 0.3},
    'infection': {'Brain with Contrast': 0.4, 'Cervical Spine with Contrast': 0.2, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine with Contrast': 0.2, 'Whole Spine with Contrast': 0.1},
    'dementia': {'Brain': 0.9, 'Brain with Contrast': 0.1},
    'aneurysm': {'Brain with Contrast': 1.0}, # Fixed: Removed 0.9+1.0 error
    'stroke': {'Brain': 0.7, 'Brain with Contrast': 0.3},
    'multiple sclerosis': {'Brain': 0.5, 'Whole Spine': 0.5},
    'back pain': {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
    'sciatica': {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
    'spinal cord injury': {'Whole Spine': 0.4, 'Whole Spine with Contrast': 0.6},
    'other': {'Brain': 0.1, 'Brain with Contrast': 0.1, 'Cervical Spine': 0.1, 'Thoracic Spine': 0.1, 'Lumbar Spine': 0.1, 'Whole Spine': 0.1}
}

# Scan Details (to check if contrast is needed)
scan_types_details = {
    'Brain': {'Contrast': False}, 'Brain with Contrast': {'Contrast': True},
    'Cervical Spine': {'Contrast': False}, 'Cervical Spine with Contrast': {'Contrast': True},
    'Thoracic Spine': {'Contrast': False}, 'Thoracic Spine with Contrast': {'Contrast': True},
    'Lumbar Spine': {'Contrast': False}, 'Lumbar Spine with Contrast': {'Contrast': True},
    'Whole Spine': {'Contrast': False}, 'Whole Spine with Contrast': {'Contrast': True}
}

clinical_indications = list(clinical_mapping.keys())

# --- 3. Generate Requests for ALL 5000 Patients ---
# We iterate through the ACTUAL patients, not random samples of them
data = []
current_date = datetime(2025, 10, 1, 8, 0, 0)

print("Generating clinical requests for all patients...")

for index, row in df_patient_details.iterrows():
    # Patient Details (Real Data)
    p_num = row['PatientNumber']
    age = row['Age']
    gender = row['Gender']
    delay_reason = row['Delay Reason']
    
    # Assign Clinical Indication (Randomly)
    indication = np.random.choice(clinical_indications)
    
    # Assign Scan Type (Based on Indication Probability)
    scan_options = clinical_mapping.get(indication, clinical_mapping['other'])
    scan_names = list(scan_options.keys())
    scan_probs = list(scan_options.values())
    
    # Normalize probabilities to ensure they sum to 1.0
    scan_probs = np.array(scan_probs)
    scan_probs /= scan_probs.sum()
    
    scan_type = np.random.choice(scan_names, p=scan_probs)
    contrast = scan_types_details.get(scan_type, {'Contrast': False})['Contrast']
    
    # Assign a Request Date (Staggered)
    current_date += timedelta(minutes=random.randint(1, 30))
    if current_date.hour >= 18:
        current_date += timedelta(days=1)
        current_date = current_date.replace(hour=8, minute=0)

    data.append([p_num, age, gender, delay_reason, indication, scan_type, contrast, current_date])

# Create the Full 5000 Request DataFrame
df_full_requests = pd.DataFrame(data, columns=[
    'Patient Number', 'Age', 'Gender', 'Delay Reason', 
    'Type of Issue', 'Scan Type', 'Contrast', 'Request Date'
])

# Save the big file (Optional, for reference)
df_full_requests.to_csv('5000_mri_requests_full.csv', index=False)
print(f"Generated 5000 requests. Saved '5000_mri_requests_full.csv'.")


# --- 4. Stratified Sampling (Select 500 for Simulation) ---
# Now we take the sample you need for the Scheduler
print("Performing stratified sampling...")

try:
    df_sampled, _ = train_test_split(
        df_full_requests, 
        train_size=500, 
        stratify=df_full_requests['Delay Reason'], 
        random_state=42
    )
    print("Stratified Sampling Successful.")
except ValueError:
    print("Warning: Stratification failed (rare classes). Using random sample.")
    df_sampled = df_full_requests.sample(n=500, random_state=42)

# Save the Final Input File for the Scheduler
df_sampled.to_csv('sampled_mri_requests_dataset.csv', index=False)
print(f"DONE. Created 'sampled_mri_requests_dataset.csv' with {len(df_sampled)} records.")
print(df_sampled['Delay Reason'].value_counts().head())