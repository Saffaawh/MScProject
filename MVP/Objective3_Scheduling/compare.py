#this is going to compare the results of the patients historical overun times vs their overrun predicted times using the average and using the ML Version
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_comparison():
    print("Loading datasets...")
    
    # 1. Load the three datasets
    # ---------------------------------------------------------
    try:
        df_ml = pd.read_csv('final_ml_schedule.csv')
        df_heuristic = pd.read_csv('final_heuristic_schedule_averages.csv')
        df_history = pd.read_csv('final_historical_mri_dataset.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the files. Details: {e}")
        return

    # 2. Standardize Column Names (Crucial for Merging)
    # ---------------------------------------------------------
    # We rename columns to be consistent (PatientID) and distinct (Duration_ML vs Duration_Actual)
    
    # ML Schedule
    df_ml = df_ml.rename(columns={
        'Patient Number': 'PatientID', 
        'Duration': 'Duration_ML', 
        'Risk_Class': 'Class_ML'
    })
    
    # Heuristic Schedule
    df_heuristic = df_heuristic.rename(columns={
        'Patient Number': 'PatientID', 
        'Duration': 'Duration_Heuristic'
    })
    
    # Historical Data (The Truth)
    df_history = df_history.rename(columns={
        'PatientNumber': 'PatientID', 
        'Actual Scan Duration': 'Duration_Actual', 
        'Class': 'Class_Actual'
    })

    # 3. Merge Datasets into one "Master Comparison" Table
    # ---------------------------------------------------------
    # First, link ML predictions to the Real History using PatientID
    merged = pd.merge(df_ml[['PatientID', 'Duration_ML', 'Class_ML']], 
                      df_history[['PatientID', 'Duration_Actual', 'Class_Actual']], 
                      on='PatientID', 
                      how='inner')

    # Next, link the Heuristic predictions to the same table
    merged = pd.merge(merged, 
                      df_heuristic[['PatientID', 'Duration_Heuristic']], 
                      on='PatientID', 
                      how='inner')

    print(f"Successfully matched {len(merged)} patients for comparison.")

    if len(merged) == 0:
        print("Error: No matching Patient IDs found. Please check your CSVs.")
        return

    # 4. Calculate Errors (The Math)
    # ---------------------------------------------------------
    # Error = Absolute difference between Predicted and Actual
    merged['Error_ML'] = (merged['Duration_ML'] - merged['Duration_Actual']).abs()
    merged['Error_Heuristic'] = (merged['Duration_Heuristic'] - merged['Duration_Actual']).abs()

    # 5. Compute Key Statistics
    # ---------------------------------------------------------
    
    # MAE: On average, how many minutes wrong are we?
    mae_ml = merged['Error_ML'].mean()
    mae_heuristic = merged['Error_Heuristic'].mean()

    # RMSE: Penalizes huge mistakes (like booking 30 mins for a 90 min scan)
    rmse_ml = np.sqrt(mean_squared_error(merged['Duration_Actual'], merged['Duration_ML']))
    rmse_heuristic = np.sqrt(mean_squared_error(merged['Duration_Actual'], merged['Duration_Heuristic']))

    # "Spot On" Accuracy: How often are we within +/- 5 minutes?
    acc_ml = (merged['Error_ML'] <= 5).mean() * 100
    acc_heuristic = (merged['Error_Heuristic'] <= 5).mean() * 100
    
    # Classification Accuracy: Did ML predict the correct Risk Class?
    # We strip whitespace just in case ("Class 1 " vs "Class 1")
    merged['Class_ML_Clean'] = merged['Class_ML'].astype(str).str.strip()
    merged['Class_Actual_Clean'] = merged['Class_Actual'].astype(str).str.strip()
    class_acc = (merged['Class_ML_Clean'] == merged['Class_Actual_Clean']).mean() * 100

    # 6. Print the Results Table
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"FINAL HEAD-TO-HEAD RESULTS (N={len(merged)} Patients)")
    print("="*60)
    print(f"{'Metric':<30} | {'Heuristic (Avg)':<15} | {'ML Model (AI)':<15}")
    print("-" * 66)
    print(f"{'Mean Error (MAE)':<30} | {mae_heuristic:<15.2f} | {mae_ml:<15.2f}")
    print(f"{'RMSE (Major Error Penalty)':<30} | {rmse_heuristic:<15.2f} | {rmse_ml:<15.2f}")
    print(f"{'Precision (+/- 5 mins)':<30} | {acc_heuristic:<15.1f}% | {acc_ml:<15.1f}%")
    print("-" * 66)
    
    # Declare Winner
    if mae_ml < mae_heuristic:
        imp = mae_heuristic - mae_ml
        print(f"WINNER: Machine Learning (Improved precision by {imp:.1f} mins/patient)")
    else:
        print("WINNER: Heuristic (Averages were better)")
    
    print("\n" + "="*60)
    print(f"ML CLASSIFICATION ACCURACY: {class_acc:.1f}%")
    print("="*60)
    
    # 7. Find a "Star Performer" Example
    # ---------------------------------------------------------
    # Where did ML beat Heuristic by the biggest margin?
    merged['Improvement'] = merged['Error_Heuristic'] - merged['Error_ML']
    best_case = merged.loc[merged['Improvement'].idxmax()]
    
    print("\n[Case Study: Biggest AI Win]")
    print(f"Patient ID:             {best_case['PatientID']}")
    print(f" - Actual Time Needed:   {best_case['Duration_Actual']} mins")
    print(f" - Heuristic Predicted:  {best_case['Duration_Heuristic']} mins (Error: {best_case['Error_Heuristic']}m)")
    print(f" - ML Predicted:         {best_case['Duration_ML']} mins (Error: {best_case['Error_ML']}m)")
    print(f" -> The ML model saved   {best_case['Improvement']} minutes of error.")

if __name__ == "__main__":
    run_comparison()