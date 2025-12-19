#step one full in my data 
import numpy as np
import pandas as pd

try:
    df= pd.read_csv('historical_mri_appointment_dataset_raw.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please ensure the dataset file is in the correct location.")
    exit()
# Calculate the target variable required for classification/regression truth.
df['Overrun Minutes'] = df['Actual Scan Duration'] - df['Scheduled Scan Time']

#step 2: featue engineering : this is following proposal table 2

#1. Mobility Needs: This inlides the issues Wheelchair user, Hoist required, Poor Mobility
conditions_mobility = (
    (df['Delay Reason'] == 'Wheelchair user') |
    (df['Delay Reason'] == 'Hoist required') | 
    (df['Delay Reason'] == 'Poor mobility')
)



df['Mobility Needs'] = np.where(conditions_mobility, 'Yes' , 'No' ) #issue where this was treated as binary so now need to add in columns for wheelchair user and hoist user 
# 1b. Wheelchair User Flag (Mid-level: Yes if Wheelchair OR Hoist)
conditions_wheelchair_flag = (
    (df['Delay Reason'] == 'Wheelchair user') |
    (df['Delay Reason'] == 'Hoist required')
)
df['Wheelchair User'] = np.where(conditions_wheelchair_flag, 'Yes', 'No')

# 1c. Hoist Required Flag (Highest level: Yes only for Hoist)
conditions_hoist_flag = (df['Delay Reason'] == 'Hoist required')
df['Hoist Required'] = np.where(conditions_hoist_flag, 'Yes', 'No')
#2. Cognitive Contions 
condtions_cognitive = (
    (df['Delay Reason'] == 'Cognitive impairment') |
    (df['Delay Reason'] == 'Learning difficulty')
)


df['Cognitive Conditions'] = np.where(condtions_cognitive, 'Yes', 'No')

#2b. Learning Difficulty Flag
conditions_learning_difficulty_flag = (df['Delay Reason'] == 'Learning difficulty')
df['Learning Difficulty'] = np.where(conditions_learning_difficulty_flag, 'Yes', 'No')

#2c. Cognitive Impairment Flag
conditions_cognitive_impairment_flag = (df['Delay Reason'] == 'Cognitive impairment')
df['Cognitive Impairment'] = np.where(conditions_cognitive_impairment_flag, 'Yes', 'No')


#3. Interpreter Required
df['Interpreter Required'] = np.where(df['Delay Reason'] == 'Interpreter required', 'Yes', 'No')

#4. Claustrophobic - Sedation needed 
df['Sedation needed'] = np.where(df['Delay Reason'] == 'Claustrophobic', 'Yes', 'No')

#5. Contrast Agent Needed
df['Contrast'] = np.where(df['Scan Type'].str.contains('Contrast', case = False), 'Yes', 'No')

#6. Diifcult Venous Access
df['Difficult IV Access'] = np.where(df['Delay Reason'] == 'DIVA', 'Yes', 'No')

#7. Hospit al Transport Required
df['Hospital Transport Required'] = np.where(df['Delay Reason'] == 'Hospital transport', 'Yes', 'No')

#Step 3: add the classifcation label
conditions_class =[
    (df['Overrun Minutes'] == 0),
    (df['Overrun Minutes'] > 0) & (df['Overrun Minutes'] <= 20),
    (df['Overrun Minutes'] > 20)
]

label_class = ['Class 1', 'Class 2', 'Class 3']
df['Class'] = np.select(conditions_class, label_class, default='Unknown')

#Step 4: Create the final dataframe and save to csv
final_df = df[[
    'PatientNumber',
    'Age', 
    'Gender',
    'Scan Type',
    'Scheduled Scan Time',
    'Arrival Time',
    'Actual Scan Duration',
    'Scheduled Date and Time',
    'Actual End Time',
    'Overrun Minutes',
    'Mobility Needs',
    'Wheelchair User',
    'Hoist Required',
    'Cognitive Conditions',
    'Learning Difficulty',
    'Cognitive Impairment',
    'Interpreter Required',
    'Sedation needed',
    'Contrast',
    'Difficult IV Access',
    'Hospital Transport Required',
    'Class'
]].copy()

# --- Save the final dataset ---
final_df.to_csv('final_historical_mri_dataset.csv', index=False)


# Display the first 5 rows and info of the final DataFrame
#print("\nFirst 5 rows of the Final Historical MRI Dataset:")
#print(final_df.head())