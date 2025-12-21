import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# 1. Load Data
# We need the training data to establish the "baseline"
try:
    df = pd.read_csv('final_historical_mri_dataset.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find 'final_historical_mri_dataset.csv'")
    exit()

# 2. Preprocess Data (Must match Training EXACTLY)
# Drop non-feature columns
X = df.drop(columns=['PatientNumber', 'Scheduled Date and Time', 'Actual End Time', 
                     'Class', 'Actual Scan Duration', 'Overrun Minutes'])

# Encode Categorical Variables
cols_to_encode = [ 
    'Gender', 'Scan Type', 'Mobility Needs', 'Wheelchair User', 'Hoist Required',
    'Poor Mobility', 'Cognitive Conditions', 'Learning Difficulty', 'Cognitive Impairment',
    'Interpreter Required', 'Sedation needed', 'Contrast', 
    'Difficult IV Access', 'Hospital Transport Required'
]

# Ensure Age is numeric (as per your final model)
X['Age'] = pd.to_numeric(X['Age'], errors='coerce').fillna(0)

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)

# 3. Load Model
print("Loading Model...")
model = joblib.load('random_forest_model_extracted.joblib')

# 4. Run SHAP Analysis
print("Calculating SHAP values... (This might take a minute)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_encoded)

# Identify the index for 'Class 3'
# Classes are usually sorted: ['Class 1', 'Class 2', 'Class 3']
class_names = list(model.classes_)
target_class = 'Class 3'
class_idx = class_names.index(target_class)

print(f"Generating plot for {target_class}...")

# 5. Generate Summary Plot
plt.figure()
shap.summary_plot(shap_values[class_idx], X_encoded, show=False)
plt.title(f"SHAP Values for {target_class} (Major Delay)")
plt.tight_layout()
plt.savefig('shap_summary_class3.png')
plt.show()

print("Success! Saved plot as 'shap_summary_class3.png'")