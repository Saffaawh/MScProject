import numpy as np
import pandas as pd
import os

# 1. Load the New Requests
input_file = 'sampled_mri_requests_dataset.csv' # Ensure this matches your file name

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit()

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} patient requests.")

# Handle missing values in 'Delay Reason' just in case
df['Delay Reason'] = df['Delay Reason'].fillna('None').astype(str)

# --- FEATURE ENGINEERING (Creating the Flags) ---

# 1. Mobility Needs
conditions_mobility = (
    (df['Delay Reason'].str.contains('Wheelchair', case=False)) |
    (df['Delay Reason'].str.contains('Hoist', case=False)) | 
    (df['Delay Reason'].str.contains('Poor mobility', case=False))
)
df['Mobility Needs'] = np.where(conditions_mobility, 'Yes' , 'No' ) 

# 1b. Wheelchair User Flag
conditions_wheelchair_flag = (
    (df['Delay Reason'].str.contains('Wheelchair', case=False)) |
    (df['Delay Reason'].str.contains('Hoist', case=False))
)
df['Wheelchair User'] = np.where(conditions_wheelchair_flag, 'Yes', 'No')

# 1c. Hoist Required Flag
df['Hoist Required'] = np.where(df['Delay Reason'].str.contains('Hoist', case=False), 'Yes', 'No')

# 1d. Poor Mobility Flag
df['Poor Mobility'] = np.where(df['Delay Reason'].str.contains('Poor mobility', case=False), 'Yes', 'No')

# 2. Cognitive Conditions
conditions_cognitive = (
    (df['Delay Reason'].str.contains('Cognitive', case=False)) |
    (df['Delay Reason'].str.contains('Learning', case=False))
)
df['Cognitive Conditions'] = np.where(conditions_cognitive, 'Yes', 'No')

# 2b. Specific Cognitive Flags
df['Learning Difficulty'] = np.where(df['Delay Reason'].str.contains('Learning', case=False), 'Yes', 'No')
df['Cognitive Impairment'] = np.where(df['Delay Reason'].str.contains('Cognitive', case=False), 'Yes', 'No')

# 3. Interpreter Required
df['Interpreter Required'] = np.where(df['Delay Reason'].str.contains('Interpreter', case=False), 'Yes', 'No')

# 4. Sedation / Claustrophobia
# Catches 'Claustrophobic' OR 'Sedation needed'
df['Sedation needed'] = np.where(df['Delay Reason'].str.contains('Claustro|Sedation', regex=True, case=False), 'Yes', 'No')

# 5. Contrast Agent Needed
df['Contrast'] = np.where(df['Scan Type'].str.contains('Contrast', case=False), 'Yes', 'No')

# 6. Difficult Venous Access
df['Difficult IV Access'] = np.where(df['Delay Reason'].str.contains('DIVA', case=False), 'Yes', 'No')

# 7. Hospital Transport Required
df['Hospital Transport Required'] = np.where(df['Delay Reason'].str.contains('Transport', case=False), 'Yes', 'No')

# --- SAVE THE FILE ---
# We keep only the columns relevant for the Scheduler + The new Flags
output_columns = [
    'Patient Number', 'Age', 'Gender', 'Scan Type', 'Delay Reason', 'Request Date', # Basic Info
    'Mobility Needs', 'Wheelchair User', 'Hoist Required', 'Poor Mobility',         # Mobility Flags
    'Cognitive Conditions', 'Learning Difficulty', 'Cognitive Impairment',          # Cognitive Flags
    'Interpreter Required', 'Sedation needed', 'Contrast',                          # Other Flags
    'Difficult IV Access', 'Hospital Transport Required'
]

# Select only existing columns (handles potential naming diffs)
cols_to_save = [c for c in output_columns if c in df.columns]

final_df = df[cols_to_save].copy()
final_df.to_csv('feature_engineered_requests_mri_dataset.csv', index=False)

print("\nSuccess! Created 'feature_engineered_requests_mri_dataset.csv'")
print("This file now contains all the flags (Hoist, Wheelchair, etc.) needed for the Scheduler.")