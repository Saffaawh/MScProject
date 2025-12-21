#this i will be running a script that uses my final MRI histotical datset to predict how much extra time a patient will need based on this data usind decision tree and random forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
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

# NOTE: We keep 'Gender' this time because it helps prediction. 
# We only drop the ID and Date columns.
X = df.drop(columns=['PatientNumber', 'Scheduled Date and Time', 'Actual End Time', 
                     'Class', 'Actual Scan Duration', 'Overrun Minutes'])

# Step 3: Encode categorical variables
# CRITICAL FIX: 'Age' is NOT in this list. It stays numeric.
# 'Gender' IS in this list.
#identify categorical columns
categorical_features = ['Gender', 'Scan Type', 
                        'Mobility Needs', 'Wheelchair User', 
                        'Hoist Required', 'Poor Mobility', 
                        'Cognitive Conditions', 'Learning Difficulty', 
                        'Cognitive Impairment', 
                        'Interpreter Required', 'Sedation needed', 
                        'Contrast', 'Difficult IV Access', 'Hospital Transport Required']

#identify numerical columns
numerical_features = ['Age']

#preprocess the data

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                       ('onehot', OneHotEncoder(handle_unknown='ignore'))])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]
                                          )
preprosessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                    ])

#build pipelines 

dt_pipeline = Pipeline(steps=[('preprocessor', preprosessor),
                    ('classifier', DecisionTreeClassifier(random_state=42))])

rf_pipeline = Pipeline(steps=[('preprocessor', preprosessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

#train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 5: Train the models
print("\nTraining Models...")

# Decision Tree Classifier
dt_pipeline.fit(X_train, y_train)

# Random Forest Classifier
rf_pipeline.fit(X_train, y_train)
# Step 6: Evaluate the models (Restored Confusion Matrix)

# Evaluate
print("\nDecision Tree Classifier:")
y_pred_dt = dt_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))

print("\nRandom Forest Classifier:")
y_pred_rf = rf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Save the pipelines
joblib.dump(dt_pipeline, 'decision_tree_model_pipeline.joblib')
joblib.dump(rf_pipeline, 'random_forest_model_pipeline.joblib')

print("\nPipeline-based models saved successfully.")

# CRITICAL: Save the feature names so the Scheduler doesn't crash!
joblib.dump(X_train.columns, 'model_feature_names.joblib')

print("\nModels and Feature Names saved successfully.")