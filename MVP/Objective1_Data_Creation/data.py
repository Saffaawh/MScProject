import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('historical_mri_appointment_dataset_raw.csv') # Fixed typo in filename
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: File not found. Please check the filename.")

# --- DEBUG STEP: Check exact spelling of reasons ---
# This helps you see if it is saved as "Hoist Required" or "Hoist required" etc.
print("\nUnique Delay Reasons found in dataset:")
print(df['Delay Reason'].unique())

# --- 1. Validation of Control Group ---
# Logic: If Delay Reason is 'None' (or 'No Delay'), Overrun MUST be 0.
# Note: Adjust 'None' if your dataset uses 'No Delay' or NaN for clean patients
control_group_errors = df[(df['Delay Reason'] == 'None') & (df['Overrun Minutes'] > 0)]
print(f"\nControl Group Validity Check: {len(control_group_errors)} errors found (Should be 0).")

# --- 2. Stochastic Variance Analysis ---
# Instead of checking a flag column == 1, we check if Delay Reason == 'Specific String'

# Lucky Hoists: Reason is 'Hoist required' but Overrun was < 20 mins
lucky_hoists = df[(df['Delay Reason'] == 'Hoist required') & (df['Overrun Minutes'] < 20)]

# Lucky Claustrophobes: Reason is 'Claustrophobic' but Overrun was < 20 mins
lucky_claustrophobes = df[(df['Delay Reason'] == 'Claustrophobic') & (df['Overrun Minutes'] < 20)]

# Lucky Transport: Reason is 'Hospital Transport' (or 'Transport required') but Overrun < 20
# CHECK THE PRINT OUTPUT ABOVE to see which string matches your data
lucky_transport = df[(df['Delay Reason'] == 'Hospital Transport') & (df['Overrun Minutes'] < 20)]

print(f"\n--- Stochastic Variance Analysis ---")
print(f"Total 'Lucky' Hoist Patients (Class 2 instead of 3): {len(lucky_hoists)}")
print(f"Total 'Lucky' Claustrophobic Patients (Class 2 instead of 3): {len(lucky_claustrophobes)}")
print(f"Total 'Lucky' Transport Patients (Class 2 instead of 3): {len(lucky_transport)}")

# Output sample for the report
if not lucky_hoists.empty:
    print("\nExample of a 'Lucky' Hoist Patient:")
    print(lucky_hoists[['PatientNumber', 'Delay Reason', 'Overrun Minutes']].head(1))