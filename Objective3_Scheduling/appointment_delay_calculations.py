#This code trains a regression model to predict appointment overruns based on patient and appointment features.
#It evaluates two models (Random Forest and Gradient Boosting) and selects the best one based on MAE.
#The final model is saved, and average predicted overruns by patient flags are calculated and saved to a CSV file.  

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer 
import joblib
import random
import datetime

#get my dataset for training regression model

try:
    df_historical = pd.read_csv('final_historical_mri_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please ensure the dataset file is in the correct location.")
    exit()
# Check for missing values


Random_seed = 42 #make this repeatable 
Ouptpu_model_name = 'appointment_delay_regression_model.pkl'
Output_flag_CSV = 'Average_predicted_Overruns_by_flag.csv'

Categorical_features = [
    'Scan Type', 'Mobility Needs', 'Wheelchair User', 'Hoist Required',
    'Cognitive Conditions', 'Learning Difficulty', 'Cognitive Impairment',
    'Interpreter Required', 'Sedation needed', 
    'Contrast', 'Difficult IV Access', 'Hospital Transport Required'
]
Numerical_features = [
    'Scheduled Scan Time', 'Age', 'Arrival Time',
]

All_Features = Categorical_features + Numerical_features




target = 'Overrun Minutes'  #target variable
features = All_Features

X= df_historical[features]
y= df_historical[target]


X_train, X_test, y_train, y_test = train_test_split(df_historical[features], df_historical[target], test_size=0.2, random_state=Random_seed)

#Preprocessing and model pipeline
#this stops NaN values causing issues

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprossor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, Numerical_features),
        ('cat', categorical_transformer, Categorical_features)
    ])

def evaluate_and_train_model(model, model_name, X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprossor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2 ): {r2}")
    
    return mae, pipeline

#train and compare models 

rfr_mae, rfr_pipeline = evaluate_and_train_model(RandomForestRegressor(n_estimators=100, random_state=Random_seed), "Random Forest Regressor (RFR)", X_train, y_train, X_test, y_test)
gbr_mae, gbr_pipeline = evaluate_and_train_model(GradientBoostingRegressor(n_estimators=100, random_state=Random_seed), "Gradient Boosting Regressor (GBR)", X_train, y_train, X_test, y_test)



if gbr_mae <= rfr_mae:
    final_model = gbr_pipeline
    print("Gradient Boosting Regressor selected as the final model.")  
else:
    final_model = rfr_pipeline  
    print("Random Forest Regressor selected as the final model.")

joblib.dump(final_model, 'appointment_delay_regression_model.pkl')



#save the model

#Extract the learned overrun minutes by flag
# List of individual flag columns (excluding Scan Type as it's not a patient flag)
individual_flags = [
    'Mobility Needs', 'Wheelchair User', 'Hoist Required',
    'Cognitive Conditions', 'Learning Difficulty', 'Cognitive Impairment',
    'Interpreter Required', 'Sedation needed', 
    'Contrast', 'Difficult IV Access', 'Hospital Transport Required'
]

all = df_historical[All_Features]
y_historical_raw = final_model.predict(all)
y_historical_predicted = np.maximum(0, y_historical_raw) #ensure no negative predictions
df_historical['Predicted Overrun Minutes'] = y_historical_predicted 

# Create Compiston Flag based on predicted overrun minutes
flag_overrun = []
for flag in individual_flags:
    flag_yes = df_historical[df_historical[flag] == 'Yes']

    if len(flag_yes) > 0:
        avg_overrun = flag_yes['Predicted Overrun Minutes'].mean()
        count = len(flag_yes)
        flag_overrun.append({
            'Flag': flag,
            'Average Predicted Overrun Minutes': round(avg_overrun, 2),
            'Number of Patients': count
        })
    else: 
        print(f"No patients with {flag} = 'Yes' found.")

    
# Create and save the summary
flag_summary_df = pd.DataFrame(flag_overrun)
flag_summary_df = flag_summary_df.sort_values('Average Predicted Overrun Minutes', ascending=False)
flag_summary_df.to_csv(Output_flag_CSV, index=False)

print(f"\nAverage Predicted Overrun Minutes by Individual Flag:")
print(flag_summary_df.to_string(index=False))
print(f"\nSaved to {Output_flag_CSV}")

# Additional: Calculate average overrun by Class as well
class_overruns = df_historical.groupby('Class')['Predicted Overrun Minutes'].agg(['mean', 'count']).round(2)
class_overruns.columns = ['Average Predicted Overrun Minutes', 'Number of Patients']
class_overruns.reset_index(inplace=True)
class_overruns.to_csv('Average_predicted_Overruns_by_Class.csv', index=False)

print(f"\nAverage Predicted Overrun Minutes by Class:")
print(class_overruns.to_string(index=False))






