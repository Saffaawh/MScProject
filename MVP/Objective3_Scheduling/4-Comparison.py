# Save this as: Scheduling_System_Comparisons.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix

def run_comparison():
    print("Loading datasets...")
    
    # 1. Load the three datasets
    # ---------------------------------------------------------
    try:
        df_ml = pd.read_csv('final_ml_schedule.csv')
        df_heuristic = pd.read_csv('final_heuristic_schedule_averages.csv')
        df_history = pd.read_csv('final_historical_mri_dataset.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Standardize Column Names (Robust Fix)
    # ---------------------------------------------------------
    
    # --- FIX FOR ML DATA ---
    # Handle "Risk_Class" (Old) vs "Predicted_Class" (New)
    if 'Predicted_Class' in df_ml.columns:
        df_ml.rename(columns={'Predicted_Class': 'Class_ML'}, inplace=True)
    elif 'Risk_Class' in df_ml.columns:
        df_ml.rename(columns={'Risk_Class': 'Class_ML'}, inplace=True)
    
    # Handle "Patient Number" vs "PatientNumber"
    if 'Patient Number' in df_ml.columns:
        df_ml.rename(columns={'Patient Number': 'PatientID'}, inplace=True)
        
    df_ml.rename(columns={'Duration': 'Duration_ML'}, inplace=True)

    # Check if we have the critical column now
    if 'Class_ML' not in df_ml.columns:
        print("CRITICAL ERROR: Could not find 'Risk_Class' or 'Predicted_Class' in ML file.")
        print("Columns found:", df_ml.columns.tolist())
        return

    # --- FIX FOR HEURISTIC DATA ---
    df_heuristic = df_heuristic.rename(columns={
        'Patient Number': 'PatientID', 
        'Duration': 'Duration_Heuristic'
    })
    
    # --- FIX FOR HISTORY DATA ---
    df_history = df_history.rename(columns={
        'PatientNumber': 'PatientID', 
        'Actual Scan Duration': 'Duration_Actual', 
        'Class': 'Class_Actual'
    })

    # 3. Merge Datasets
    # ---------------------------------------------------------
    print("Merging data...")
    merged = pd.merge(df_ml[['PatientID', 'Duration_ML', 'Class_ML']], 
                      df_history[['PatientID', 'Duration_Actual', 'Class_Actual']], 
                      on='PatientID', 
                      how='inner')

    merged = pd.merge(merged, 
                      df_heuristic[['PatientID', 'Duration_Heuristic']], 
                      on='PatientID', 
                      how='inner')

    print(f"Successfully matched {len(merged)} patients for comparison.")

    if len(merged) == 0:
        print("Error: No matching Patient IDs found. Check if your datasets cover the same patients.")
        return

    # 4. Calculate Regression Errors (The Math)
    # ---------------------------------------------------------
    merged['Error_ML'] = (merged['Duration_ML'] - merged['Duration_Actual']).abs()
    merged['Error_Heuristic'] = (merged['Duration_Heuristic'] - merged['Duration_Actual']).abs()

    mae_ml = merged['Error_ML'].mean()
    mae_heuristic = merged['Error_Heuristic'].mean()
    
    # Root Mean Squared Error (Penalizes big mistakes more)
    rmse_ml = np.sqrt(mean_squared_error(merged['Duration_Actual'], merged['Duration_ML']))
    rmse_heuristic = np.sqrt(mean_squared_error(merged['Duration_Actual'], merged['Duration_Heuristic']))
    
    # Precision within 5 mins
    acc_ml = (merged['Error_ML'] <= 5).mean() * 100
    acc_heuristic = (merged['Error_Heuristic'] <= 5).mean() * 100
    
    # 5. Prepare Classification Data (The Logic)
    # ---------------------------------------------------------
    # Clean strings to ensure matching (e.g. "Class 1" vs "Class 1 ")
    merged['Class_ML_Clean'] = merged['Class_ML'].astype(str).str.strip()
    merged['Class_Actual_Clean'] = merged['Class_Actual'].astype(str).str.strip()

    # Define the order of labels
    labels = ['Class 1', 'Class 2', 'Class 3']

    # Generate Metrics
    report = classification_report(
        merged['Class_Actual_Clean'], 
        merged['Class_ML_Clean'], 
        labels=labels,
        zero_division=0
    )
    
    cm = confusion_matrix(
        merged['Class_Actual_Clean'], 
        merged['Class_ML_Clean'], 
        labels=labels
    )

    # 6. Print the Regression Results Table
    # ---------------------------------------------------------
    print("\n" + "="*66)
    print(f"PART 1: REGRESSION EFFICIENCY (N={len(merged)})")
    print("="*66)
    print(f"{'Metric':<30} | {'Heuristic (Avg)':<15} | {'ML Model (AI)':<15}")
    print("-" * 66)
    print(f"{'Mean Error (MAE)':<30} | {mae_heuristic:<15.2f} | {mae_ml:<15.2f}")
    print(f"{'RMSE (Major Error Penalty)':<30} | {rmse_heuristic:<15.2f} | {rmse_ml:<15.2f}")
    print(f"{'Precision (+/- 5 mins)':<30} | {acc_heuristic:<15.1f}% | {acc_ml:<15.1f}%")
    print("-" * 66)
    
    if mae_ml < mae_heuristic:
        imp = mae_heuristic - mae_ml
        print(f"WINNER: Machine Learning (Improved precision by {imp:.1f} mins/patient)")
    else:
        print("WINNER: Heuristic (Averages were better)")
    
    # 7. Print the Detailed Classification Results
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("PART 2: CLASSIFICATION PERFORMANCE (Precision/Recall)")
    print("="*60)
    print("Detailed Classification Report:\n")
    print(report)
    
    print("-" * 60)
    print("Confusion Matrix (Actual vs Predicted):")
    print(f"{'':<15} | {'Pred: Class 1':<13} | {'Pred: Class 2':<13} | {'Pred: Class 3':<13}")
    print("-" * 60)
    
    row_names = ['Actual: Class 1', 'Actual: Class 2', 'Actual: Class 3']
    for i, row in enumerate(cm):
        print(f"{row_names[i]:<15} | {row[0]:<13} | {row[1]:<13} | {row[2]:<13}")
        
    print("\nNOTE: Focus on Class 3 Recall (Sensitivity). This tells you what % of")
    print("Major Delays were successfully caught by the model.")

    # 8. Find a "Star Performer" Example
    # ---------------------------------------------------------
    merged['Improvement'] = merged['Error_Heuristic'] - merged['Error_ML']
    best_case = merged.loc[merged['Improvement'].idxmax()]
    
    print("\n" + "="*60)
    print("[Case Study: Biggest AI Win]")
    print(f"Patient ID:             {best_case['PatientID']}")
    print(f" - Actual Time Needed:   {best_case['Duration_Actual']} mins")
    print(f" - Heuristic Predicted:  {best_case['Duration_Heuristic']} mins (Error: {best_case['Error_Heuristic']}m)")
    print(f" - ML Predicted:         {best_case['Duration_ML']} mins (Error: {best_case['Error_ML']}m)")
    print(f" -> The ML model saved   {best_case['Improvement']} minutes of error.")

if __name__ == "__main__":
    run_comparison()