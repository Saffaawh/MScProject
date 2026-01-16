import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- CONFIGURATION ---
DATA_FILE = 'final_historical_mri_dataset.csv'
MODEL_FILE = 'random_forest_model_pipeline.joblib' 

# UPDATED LIST: Removed Aggregates (Mobility Needs, Cognitive Impairment) and Contrast
CLINICAL_FLAGS = [
    'Wheelchair User', 
    'Hoist Required', 
    'Poor Mobility', 
    'Cognitive Conditions', 
    'Learning Difficulty', 
    'Interpreter Required', 
    'Sedation needed', 
    'Difficult IV Access', 
    'Hospital Transport Required'
]

def analyze_clinical_errors():
    print("1. Loading Data and Model...")
    try:
        df = pd.read_csv(DATA_FILE)
        pipeline = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("Error: Files not found. Check your paths.")
        return

    # 2. Get Model Predictions
    target_col = 'Class'
    drop_cols = ['PatientNumber', 'Scheduled Date and Time', 'Actual End Time', 
                 'Actual Scan Duration', 'Overrun Minutes', target_col]
    X_pred = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    print("2. Predicting and Categorizing Errors...")
    df['Predicted_Class'] = pipeline.predict(X_pred)
    
    # Define the 4 Outcome Groups
    def get_outcome_group(row):
        t = row[target_col]
        p = row['Predicted_Class']
        
        if t == 'Class 2' and p == 'Class 2':
            return 'Correct Class 2'
        elif t == 'Class 3' and p == 'Class 3':
            return 'Correct Class 3'
        elif t == 'Class 3' and p == 'Class 2':
            return 'ERROR: Actual 3 -> Pred 2 (Underestimated)'
        elif t == 'Class 2' and p == 'Class 3':
            return 'ERROR: Actual 2 -> Pred 3 (Overestimated)'
        else:
            return 'Other' # Class 1 interactions or other mismatches

    df['Outcome_Group'] = df.apply(get_outcome_group, axis=1)

    # 3. Aggregating Data by Flag
    print("3. Analyzing each specific clinical flag...")
    
    summary_data = []
    
    for flag in CLINICAL_FLAGS:
        if flag not in df.columns:
            print(f"   ! Warning: Column '{flag}' not found. Skipping.")
            continue
            
        # Normalize to find "Yes" / "True" / "1"
        mask_present = df[flag].astype(str).str.lower().isin(['yes', '1', 'true', 'y'])
        subset = df[mask_present]
        
        if len(subset) == 0:
            print(f"   ! Warning: No patients found with flag '{flag}'.")
            continue
            
        counts = subset['Outcome_Group'].value_counts()
        
        row = {
            'Flag': flag,
            'Total Patients': len(subset),
            'Correct Class 2': counts.get('Correct Class 2', 0),
            'Correct Class 3': counts.get('Correct Class 3', 0),
            # This is the dangerous error (Underestimation)
            'ERROR: Actual 3 -> Pred 2': counts.get('ERROR: Actual 3 -> Pred 2 (Underestimated)', 0),
            # This is the inefficiency error (Overestimation)
            'ERROR: Actual 2 -> Pred 3': counts.get('ERROR: Actual 2 -> Pred 3 (Overestimated)', 0)
        }
        summary_data.append(row)

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Flag')
    
    # Calculate "Underestimation Rate" (Safety Risk) for sorting
    summary_df['Safety_Risk_Rate'] = summary_df['ERROR: Actual 3 -> Pred 2'] / summary_df['Total Patients']
    summary_df = summary_df.sort_values('Safety_Risk_Rate', ascending=True)

    print("\n--- Summary Table (Sorted by Safety Risk) ---")
    print(summary_df[['Total Patients', 'Safety_Risk_Rate', 'ERROR: Actual 3 -> Pred 2']])

    # 4. Plotting
    print("4. Generating Visualization...")
    
    # Plot only the 4 relevant outcome columns
    plot_cols = ['Correct Class 2', 'Correct Class 3', 
                 'ERROR: Actual 3 -> Pred 2', 'ERROR: Actual 2 -> Pred 3']
    
    colors = ['skyblue', 'salmon', 'blue', 'red']
    
    plt.figure(figsize=(14, 10))
    summary_df[plot_cols].plot(
        kind='barh', 
        stacked=True, 
        color=colors, 
        figsize=(12, 8),
        width=0.8
    )
    
    plt.title('Prediction Outcomes by Specific Clinical Need\n(Dark Blue = Dangerous Underestimation)', fontsize=16)
    plt.xlabel('Number of Patients', fontsize=12)
    plt.ylabel('Clinical Flag', fontsize=12)
    plt.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_filename = 'clinical_flag_specific_analysis.png'
    plt.savefig(output_filename)
    print(f"Done! Chart saved to '{output_filename}'")

if __name__ == "__main__":
    analyze_clinical_errors()