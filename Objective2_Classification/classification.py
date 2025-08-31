#this i will be running a script that uses my final MRI histotical datset to predict how much extra time a patient will need based on this data usind decision tree and random forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#Step 1: Load the dataset
try:
    df = pd.read_csv('final_historical_mri_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please ensure the dataset file is in the correct location.")
    exit()

#2 define features and label
y= df['Class']

X = df.drop(columns=['PatientNumber', 'Gender','Scheduled Date and Time', 'Actual End Time', 'Class', 'Actual Scan Duration'] )#dropping unneccessary columns

#3. encode categorical variables
X = pd.get_dummies(X, columns=[ 'Scan Type', 
                                'Age', 
                                'Mobility Needs', 
                                'Cognitive Conditions', 
                                'Interpreter Required', 
                                'Sedation needed', 
                                'Contrast', 
                                'Difficult IV Access', 
                                'Hospital Transport Required'
                                 ], drop_first=True)
print("\nFeatures and target variable defined. Categorical data encoded.")
print(f"Shape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")



#Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)    
#Step 4: Train the models
#Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
#Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
#Step 5: Evaluate the models
#Decision Tree Evaluation
y_pred_dt = dt_classifier.predict(X_test)
print("Decision Tree Classifier Report:")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
#Random Forest Evaluation
y_pred_rf = rf_classifier.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
#Step 6: Save the models
joblib.dump(dt_classifier, 'decision_tree_model.joblib')
joblib.dump(rf_classifier, 'random_forest_model.joblib')
print("Models saved successfully.")

