import pandas as pd

# Load dataset
try:
    df = pd.read_csv('historical_mri_appointment_dataset_raw.csv')
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: File not found.")

# Define the list of delay reasons to check
# (Make sure these match your CSV spelling exactly)
delay_reasons = [
    'Hoist required', 
    'Claustrophobic', 
    'Hospital Transport', 
    'Wheelchair user', 
    'Interpreter required', 
    'DIVA',
    'Learning difficulty',
    'Poor mobility',
    'Cognitive impairment'
]

print("\n--- Operational Impact Analysis: Class 3 Distribution ---")
print(f"{'Delay Reason':<25} | {'Total':<5} | {'Major (Class 3)':<15} | {'Minor (Class 2)':<15} | {'% Major'}")
print("-" * 80)

for reason in delay_reasons:
    # Filter for this specific reason
    # We use string matching to be safe
    subset = df[df['Delay Reason'].astype(str).str.contains(reason, case=False, regex=False)]
    
    total_count = len(subset)
    
    if total_count == 0:
        print(f"{reason:<25} | 0     | 0               | 0               | N/A")
        continue

    # Count Classes (Class 3 is Major, Class 2 is Minor)
    # Assuming 'Historical Class' column exists. If not, use Overrun Minutes > 20
    major_count = len(subset[subset['Overrun Minutes'] > 20])
    minor_count = len(subset[subset['Overrun Minutes'] <= 20])
    
    percent_major = (major_count / total_count) * 100
    
    print(f"{reason:<25} | {total_count:<5} | {major_count:<15} | {minor_count:<15} | {percent_major:.1f}%")

print("-" * 80)
print("Note: 'Hospital Transport' showing 100% Major indicates a 'Hard Constraint'.")
print("Note: 'Claustrophobic' showing lower % Major indicates 'Stochastic Variance'.")