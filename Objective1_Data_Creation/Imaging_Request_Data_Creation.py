# SCRIPT 1: GENERATE THE FULL 5000-REQUEST POOL (Simplified with Print Statements)

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import sys
from sklearn.model_selection import train_test_split # Still imported, but not used

# --- Definitions ---
scan_types_details = {
    'Brain': {'ScheduledTime': 30, 'Contrast' : False},
    'Brain with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Cervical Spine': {'ScheduledTime': 30, 'Contrast' : False},
    'Cervical Spine with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Thoracic Spine': {'ScheduledTime': 30, 'Contrast' : False},
    'Thoracic Spine with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Lumbar Spine': {'ScheduledTime': 30, 'Contrast' : False}, 
    'Lumbar Spine with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Whole Spine': {'ScheduledTime': 45, 'Contrast' : False},
    'Whole Spine with Contrast': {'ScheduledTime': 60, 'Contrast' : True}}

clinincal_mapping = { 
    'headache': {'Brain': 0.8, 'Brain with Contrast': 0.2},
    'neck pain': {'Cervical Spine': 0.8, 'Cervical Spine with Contrast': 0.2},
    'tumor': {'Brain with Contrast': 0.4, 'Cervical Spine with Contrast': 0.1, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine with Contrast': 0.1, 'Whole Spine with Contrast': 0.3},
    'infection': {'Brain with Contrast': 0.4, 'Cervical Spine with Contrast': 0.2, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine with Contrast': 0.2, 'Whole Spine with Contrast': 0.1},
    'dementia': {'Brain': 0.9, 'Brain with Contrast': 0.1},
    'aneurysm': {'Brain with Contrast': 1.0, 'Brain': 0.9},
    'stroke': {'Brain': 0.7, 'Brain with Contrast': 0.3},
    'multiple sclerosis': {'Brain': 0.5, 'Whole Spine': 0.5},
    'back pain': {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
    'sciatica': {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
    'spinal cord injury': {'Whole Spine': 0.4, 'Whole Spine with Contrast': 0.6},
    'other': {'Brain': 0.1, 'Brain with Contrast': 0.1, 'Cervical Spine': 0.1, 'Cervical Spine with Contrast': 0.1, 
              'Thoracic Spine': 0.1, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine': 0.1, 
              'Lumbar Spine with Contrast': 0.1, 'Whole Spine': 0.1, 'Whole Spine with Contrast': 0.1}
}

clinical_indications = list(clinincal_mapping.keys())

type_of_problem = { 
    'headache': 0.25, 'neck pain': 0.15, 'tumor': 0.1, 'infection': 0.1, 
    'dementia': 0.1, 'aneurysm': 0.05, 'stroke': 0.05, 'multiple sclerosis': 0.05, 
    'back pain': 0.1, 'sciatica': 0.025, 'spinal cord injury': 0.025, 'other': 0.05
}

# --- Probability Calculations (Fixes TypeErrors) ---
type_of_problem_names = list(type_of_problem.keys())
type_of_problem_probs = [float(p) for p in type_of_problem.values()] 
type_of_problem_cumprobs = np.cumsum(type_of_problem_probs)

def select_scan_type(clinical_indication):
    scan_type_probs = clinincal_mapping[clinical_indication]
    scan_types = list(scan_type_probs.keys())
    probs = list(scan_type_probs.values())
    if not np.isclose(sum(probs), 1.0):
        probs = np.array(probs) / sum(probs)
    cum_probs = np.cumsum(probs)
    rand_val = random.random()
    for i, cp in enumerate(cum_probs):
        if rand_val < cp:
            return scan_types[i]
    return scan_types[-1]

FLAG_COLUMNS = [
    'Mobility Needs', 'Cognitive Conditions', 'Interpreter Required', 
    'Sedation needed', 'Difficult IV Access', 'Hospital Transport Required'
]
OUTPUT_FILE_NAME = 'raw_5000_mri_requests_pool.csv' 


# --- Data Loading and Selection ---

try:
    df_historical = pd.read_csv('final_historical_mri_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please ensure the dataset file is in the correct location.")
    sys.exit(1)

COLS_TO_KEEP = ['PatientNumber', 'Age', 'Gender', 'Class'] + FLAG_COLUMNS
try:
    # Iterate over all 5000 records
    df_patients_for_requests = df_historical[COLS_TO_KEEP].reset_index(drop=True)
    print(f"Preparing to generate requests for all {len(df_patients_for_requests)} historical patients.")
except KeyError as e:
    print(f"Missing required column in historical data: {e}. Please check historical column names.")
    sys.exit(1)


# --- Request Generation Loop ---

imaging_requests = []
for index, row in df_patients_for_requests.iterrows():
    
    patient_number = row['PatientNumber']
    age = row['Age']
    gender = row['Gender']
    patient_class = row['Class'] 
    
    # Collect Patient Flags
    mobility_needs = row['Mobility Needs']
    cognitive_conditions = row['Cognitive Conditions']
    interpreter_required = row['Interpreter Required']
    sedation_needed = row['Sedation needed']
    difficult_iv_access = row['Difficult IV Access']
    hospital_transport_required = row['Hospital Transport Required']
    
    # Randomly assign a Clinical Indication (Synthetic Request)
    clinical_indication = 'other' 
    rand_val = random.random()
    for i, cp in enumerate(type_of_problem_cumprobs):
        if rand_val < cp:
            clinical_indication = type_of_problem_names[i]
            break
            
    # Select a scan type based on the clinical indication
    scan_type = select_scan_type(clinical_indication)
    scheduled_time = scan_types_details[scan_type]['ScheduledTime']
    requires_contrast = scan_types_details[scan_type]['Contrast']

    # Append all data points to the list
    imaging_requests.append({
        'RequestID': f'R{index+1}',
        'PatientNumber': patient_number,
        'Age': age,
        'Gender': gender,
        
        'Mobility Needs': mobility_needs,
        'Cognitive Conditions': cognitive_conditions,
        'Interpreter Required': interpreter_required,
        'Sedation needed': sedation_needed,
        'Difficult IV Access': difficult_iv_access,
        'Hospital Transport Required': hospital_transport_required,
        
        'Clinical Indication': clinical_indication,
        'Scan Type': scan_type,
        'Scheduled Time (min)': scheduled_time,
        'Requires Contrast': requires_contrast,
        'Historical Class': patient_class 
    })

# Convert to DataFrame and save
df_large_requests = pd.DataFrame(imaging_requests)
df_large_requests.to_csv(OUTPUT_FILE_NAME, index=False)
print(f"Step 1 Complete: Large request pool saved to {OUTPUT_FILE_NAME}.")