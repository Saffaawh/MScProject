import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Step 1: Load the dataset
try:
    df = pd.read_csv('final_historical_mri_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'final_historical_mri_dataset.csv' not found.")
    exit()

# Step 2: Define features and label
y = df['Class']

# Drop non-feature columns
X = df.drop(columns=['PatientNumber', 'Scheduled Date and Time', 'Actual End Time', 
                     'Class', 'Actual Scan Duration', 'Overrun Minutes'])

# Step 3: Define Column Groups
categorical_features = [
    'Gender', 'Scan Type', 'Mobility Needs', 'Wheelchair User', 
    'Hoist Required', 'Poor Mobility', 'Cognitive Conditions', 
    'Learning Difficulty', 'Cognitive Impairment', 
    'Interpreter Required', 'Sedation needed', 
    'Contrast', 'Difficult IV Access', 'Hospital Transport Required'
]

numerical_features = ['Age']

# Step 4: Preprocessing Pipelines

# --- FIX: Age should be Scaled, not One-Hot Encoded ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fill missing age with median
    ('scaler', StandardScaler())                   # Standardize age
])

# Categorical: Impute missing with 'most_frequent' then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Build Model Pipelines

# --- Decision Tree Configuration ---
# Note: No 'n_estimators' because it's a single tree.
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=10,               
        class_weight='balanced',    
        random_state=42             
    ))
])

# --- Random Forest Configuration ---
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,           
        max_depth=10,               
        class_weight='balanced',    
        random_state=42             
    ))
])

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Models...")
dt_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Step 7: Evaluate
print("\n--- Decision Tree Classifier ---")
y_pred_dt = dt_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_dt))

print("\n--- Random Forest Classifier ---")
y_pred_rf = rf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Step 8: Save EVERYTHING
print("\nSaving Models...")

# 1. Save Full Pipelines 
joblib.dump(dt_pipeline, 'decision_tree_model_pipeline.joblib')
joblib.dump(rf_pipeline, 'random_forest_model_pipeline.joblib')

# 2. Extract and Save Raw Models 
extracted_dt = dt_pipeline.named_steps['classifier']
joblib.dump(extracted_dt, 'decision_tree_model.joblib')

extracted_rf = rf_pipeline.named_steps['classifier']
joblib.dump(extracted_rf, 'random_forest_model_extracted.joblib')

# 3. Save Preprocessor 
extracted_preprocessor = rf_pipeline.named_steps['preprocessor']
joblib.dump(extracted_preprocessor, 'model_preprocessor.joblib')

# 4. Save Feature Names 
joblib.dump(X_train.columns, 'model_feature_names.joblib')

print("\nSUCCESS: All pipelines and extracted models saved.")

#visaulise tree 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. Get the preprocessor and the classifier from the pipeline
model = rf_pipeline.named_steps['classifier']
preprocessor = rf_pipeline.named_steps['preprocessor']

# 2. Get feature names (Complex because of OneHotEncoder)
# We need to get names from the numeric scaler AND the categorical encoder
feature_names = numerical_features + list(preprocessor.named_transformers_['cat']
                                       .named_steps['onehot']
                                       .get_feature_names_out(categorical_features))

# 3. Extract one single tree (e.g., the first one)
one_tree = model.estimators_[0]

# 4. Plot it
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(one_tree,
          feature_names=feature_names,
          class_names=['No Delay', 'Minor', 'Major'],
          filled=True,
          max_depth=3,  # Limit depth so it is readable in the report
          fontsize=10)
plt.title("Visualisation of Decision Logic (Single Tree extracted from Random Forest)")
plt.savefig('single_tree_viz.png')
print("Tree image saved as 'single_tree_viz.png'")