import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- FILES ---
INPUT_HISTORICAL = 'final_historical_mri_dataset.csv'
OUTPUT_5000 = '5000_mri_requests_full.csv'
OUTPUT_SAMPLE = 'sampled_mri_requests_dataset.csv'
# Also saving as 'feature_engineered' to match your pipeline
OUTPUT_FE = 'feature_engineered_requests_mri_dataset.csv'

# --- CONFIGURATION ---
# 1. Target Quotas for specific complex needs
QUOTAS = {
    'Hoist Required': 10,
    'Wheelchair User': 70,
    'Hospital Transport Required': 35,
    'Sedation needed': 15,
    'Interpreter Required': 25,
    'Cognitive Impairment': 20,
    'Learning Difficulty': 10,
    'Difficult IV Access': 10,
    'Poor Mobility': 15
}

# 2. Target for Standard Patients (No Flags)
TARGET_NO_FLAG = 200 

# Clinical Logic
clinical_mapping = { 
    'headache': {'Brain': 0.8, 'Brain with Contrast': 0.2},
    'neck pain': {'Cervical Spine': 0.8, 'Cervical Spine with Contrast': 0.2},
    'tumor': {'Brain with Contrast': 0.4, 'Whole Spine with Contrast': 0.3, 'Lumbar Spine with Contrast': 0.3},
    'infection': {'Brain with Contrast': 0.4, 'Whole Spine with Contrast': 0.6},
    'dementia': {'Brain': 0.9, 'Brain with Contrast': 0.1},
    'aneurysm': {'Brain with Contrast': 1.0},
    'stroke': {'Brain': 0.7, 'Brain with Contrast': 0.3},
    'multiple sclerosis': {'Brain': 0.5, 'Whole Spine': 0.5},
    'back pain': {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
    'sciatica': {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
    'spinal cord injury': {'Whole Spine': 0.4, 'Whole Spine with Contrast': 0.6},
    'other': {'Brain': 0.2, 'Lumbar Spine': 0.4, 'Cervical Spine': 0.4}
}
clinical_indications = list(clinical_mapping.keys())

def run_pipeline():
    print("--- STEP 1: LOAD & PREPARE POOL ---")
    try:
        df_hist = pd.read_csv(INPUT_HISTORICAL)
    except FileNotFoundError:
        print(f"Error: {INPUT_HISTORICAL} not found. Please upload it.")
        return

    # Standardize Patient ID
    if 'PatientNumber' in df_hist.columns:
        df_hist.rename(columns={'PatientNumber': 'Patient Number'}, inplace=True)

    # Define Flag Columns to check
    flag_cols = [
        'Hoist Required', 'Wheelchair User', 'Hospital Transport Required', 
        'Sedation needed', 'Interpreter Required', 'Poor Mobility', 
        'Cognitive Impairment', 'Learning Difficulty', 'Difficult IV Access',
        'Mobility Needs', 'Cognitive Conditions'
    ]
    flag_cols = [c for c in flag_cols if c in df_hist.columns]
    
    # Select columns to keep
    base_cols = ['Patient Number', 'Age', 'Gender']
    keep_cols = [c for c in base_cols + flag_cols if c in df_hist.columns]
    
    df_req = df_hist[keep_cols].copy()
    
    # Create 'Flag Present' Column
    def check_flag(row):
        for col in flag_cols:
            val = str(row.get(col, '')).lower()
            if val == 'yes': return 'Yes'
        return 'No'

    df_req['Flag Present'] = df_req.apply(check_flag, axis=1)
    
    print("Generating new appointment details...")
    new_indications, new_scans, new_contrasts, new_dates = [], [], [], []
    current_date = datetime(2025, 10, 1, 8, 0)

    for _ in range(len(df_req)):
        indication = np.random.choice(clinical_indications)
        scan_opts = clinical_mapping.get(indication, clinical_mapping['other'])
        probs = np.array(list(scan_opts.values()))
        probs /= probs.sum()
        scan_type = np.random.choice(list(scan_opts.keys()), p=probs)
        
        new_indications.append(indication)
        new_scans.append(scan_type)
        new_contrasts.append('Yes' if 'Contrast' in scan_type else 'No')
        
        current_date += timedelta(minutes=np.random.randint(2, 15))
        if current_date.hour >= 18:
            current_date += timedelta(days=1)
            current_date = current_date.replace(hour=8, minute=0)
        new_dates.append(current_date)

    df_req['Clinical Indication'] = new_indications
    df_req['Scan Type'] = new_scans
    df_req['Contrast'] = new_contrasts
    df_req['Request Date'] = new_dates

    # Save the full pool
    df_req.to_csv(OUTPUT_5000, index=False)
    print(f"Saved {OUTPUT_5000} ({len(df_req)} rows).")

    print("\n--- STEP 2: STRATIFIED SAMPLING ---")
    sampled_indices = []
    df_pool = df_req.copy()

    # A. Sample Specific Complex Quotas
    for col, count in QUOTAS.items():
        if col in df_pool.columns:
            available = df_pool[df_pool[col] == 'Yes']
            n_take = min(len(available), count)
            if n_take > 0:
                selected = available.sample(n=n_take, random_state=42)
                sampled_indices.extend(selected.index.tolist())
                df_pool = df_pool.drop(selected.index)
                print(f" - Added {n_take} patients for '{col}'")
    
    # B. Sample 'No Flag' Patients (Standard Class 1)
    available_no_flag = df_pool[df_pool['Flag Present'] == 'No']
    n_std_take = min(len(available_no_flag), TARGET_NO_FLAG)
    
    if n_std_take > 0:
        selected_std = available_no_flag.sample(n=n_std_take, random_state=42)
        sampled_indices.extend(selected_std.index.tolist())
        df_pool = df_pool.drop(selected_std.index)
        print(f" - Added {n_std_take} patients with Flag Present = No")

    # C. Fill Remaining Spots to reach 500
    needed = 500 - len(sampled_indices)
    if needed > 0:
        print(f" - Filling remaining {needed} slots with random patients...")
        fillers = df_pool.sample(n=needed, random_state=42)
        sampled_indices.extend(fillers.index.tolist())

    # Finalize
    df_final = df_req.loc[sampled_indices].copy()
    
    # Save Output
    df_final.to_csv(OUTPUT_SAMPLE, index=False)
    df_final.to_csv(OUTPUT_FE, index=False)
    
    print("\nSUCCESS!")
    print(f"Sampled Dataset: {len(df_final)} rows")
    print("Breakdown of 'Flag Present':")
    print(df_final['Flag Present'].value_counts())

if __name__ == "__main__":
    run_pipeline()