import pandas as pd
import numpy as np

# 1. Load the full 5000 dataset
try:
    df_requests = pd.read_csv('5000_mri_requests_full.csv')
    print("Full dataset loaded.")
except FileNotFoundError:
    print("Error: '5000_mri_requests_full.csv' not found.")
    exit()

# 2. Define your "Test Targets" (Minimum number of patients per flag)
# This forces over-representation of rare classes
target_quotas = {
    'Hoist required': 10,        # Force 10 (vs natural ~2)
    'Learning difficulty': 10,   # Force 10 (vs natural ~3)
    'DIVA': 10,                  # Force 10
    'Claustrophobic': 15,        # Force 15
    'Cognitive impairment': 20,  # Force 20
    'Interpreter required': 25,  # Force 25
    'Hospital Transport': 35,    # Force 35
    'Wheelchair user': 70,       # Force 70
    'Poor mobility': 15,         # Force 15
    'None': 0                    # We will fill the rest with 'None'
}

sampled_indices = []
df_pool = df_requests.copy()

print("Selecting specific quotas...")

# 3. Select the "Must Haves" first
for flag, target in target_quotas.items():
    if target > 0:
        # Find all patients with this specific flag
        available_patients = df_pool[df_pool['Delay Reason'] == flag]
        
        # If we have fewer than the target, take them all. Otherwise take the target count.
        n_to_take = min(len(available_patients), target)
        
        if n_to_take > 0:
            selected = available_patients.sample(n=n_to_take, random_state=42)
            sampled_indices.extend(selected.index.tolist())
            
            # Remove these from the pool so we don't pick them again
            df_pool = df_pool.drop(selected.index)
            print(f" - Added {n_to_take} patients with '{flag}'")

# 4. Fill the remaining spots to reach 500
current_count = len(sampled_indices)
needed = 500 - current_count

if needed > 0:
    print(f"Filling remaining {needed} slots with random patients...")
    # Fill primarily with standard patients or whatever is left
    fillers = df_pool.sample(n=needed, random_state=42)
    sampled_indices.extend(fillers.index.tolist())

# 5. Create and Save the Final Sample
df_final_sample = df_requests.loc[sampled_indices]
df_final_sample.to_csv('sampled_mri_requests_dataset.csv', index=False)

print("\nSUCCESS: Created 'sampled_mri_requests_dataset.csv'")
print("-" * 30)
print("Final Flag Distribution:")
print(df_final_sample['Delay Reason'].value_counts())
print("-" * 30)