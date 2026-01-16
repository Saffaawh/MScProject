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

# --- CONFIGURATION ---
RANDOM_SEED = 42
OUTPUT_BEST_MODEL = 'appointment_delay_regression_model.pkl' # Keeps Scheduler working
OUTPUT_FLAG_CSV = 'Average_predicted_Overruns_by_flag.csv'

# Step 1: Load Data
try:
    df_historical = pd.read_csv('final_historical_mri_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'final_historical_mri_dataset.csv' not found.")
    exit()

# Step 2: Define Features
# Correctly separating Age (Numeric) from Flags (Categorical)
categorical_features = [
    'Scan Type', 'Wheelchair User', 'Hoist Required','Poor Mobility', 
    'Learning Difficulty', 'Cognitive Impairment', 'Interpreter Required', 
    'Sedation needed', 'Difficult IV Access', 'Hospital Transport Required', 'Contrast'
]
numerical_features = ['Age']

target = 'Overrun Minutes'
features = numerical_features + categorical_features

X = df_historical[features]
y = df_historical[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Step 3: Define Preprocessing
# Note: Age is correctly handled here 
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 4: Training Function
def train_and_evaluate(model, name):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R2: {r2:.4f}")
    
    return pipeline, mae

# Step 5: Train Both Models

# --- Random Forest Configuration ---
rfr_pipeline, rfr_mae = train_and_evaluate(
    RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,      
        random_state=RANDOM_SEED
    ), 
    "Random Forest"
)

# --- Gradient Boosting Configuration ---
gbr_pipeline, gbr_mae = train_and_evaluate(
    GradientBoostingRegressor(
        n_estimators=100, 
        max_depth=5,        # Shallow trees for boosting
        learning_rate=0.1,  # Standard learning rate
        random_state=RANDOM_SEED
    ), 
    "Gradient Boosting"
)

# Step 6: Save Both Specific Models
print("\nSaving individual models...")
joblib.dump(rfr_pipeline, 'random_forest_regression.joblib')
joblib.dump(gbr_pipeline, 'gradient_boosting_regression.joblib')

# Step 7: Select and Save "Best" Model 
if gbr_mae <= rfr_mae:
    final_model = gbr_pipeline
    print(f"\nWINNER: Gradient Boosting (MAE {gbr_mae:.2f})")
else:
    final_model = rfr_pipeline
    print(f"\nWINNER: Random Forest (MAE {rfr_mae:.2f})")

# Save as the generic name Scheduler expects
joblib.dump(final_model, OUTPUT_BEST_MODEL)
print(f"Saved best model as '{OUTPUT_BEST_MODEL}'")

# Save feature names for future analysis
joblib.dump(features, 'regression_feature_names.joblib')

# Step 8: Generate Heuristic CSV (Using Best Model)
print("\nGenerating Heuristic Averages Table...")
data = df_historical[features]
y_pred_all = final_model.predict(data)
df_historical['Predicted Overrun Minutes'] = np.maximum(0, y_pred_all) # No negative overruns

flag_overrun = []
# Calculate averages for each flag individually
check_flags = [f for f in categorical_features if f != 'Scan Type']

for flag in check_flags:
    if flag in df_historical.columns:
        subset = df_historical[df_historical[flag] == 'Yes']
        if not subset.empty:
            avg = subset['Predicted Overrun Minutes'].mean()
            flag_overrun.append({
                'Flag': flag,
                'Average Predicted Overrun Minutes': round(avg, 2),
                'Number of Patients': len(subset)
            })

flag_summary_df = pd.DataFrame(flag_overrun)
flag_summary_df = flag_summary_df.sort_values('Average Predicted Overrun Minutes', ascending=False)
flag_summary_df.to_csv(OUTPUT_FLAG_CSV, index=False)

print(f"Success! Heuristic file saved to '{OUTPUT_FLAG_CSV}'")