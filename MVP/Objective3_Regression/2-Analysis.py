#this will analyse the 2 models against each other 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = 'final_historical_mri_dataset.csv'
RANDOM_SEED = 42

MODEL_FILES = {
    'Random Forest': 'random_forest_regression.joblib',
    'Gradient Boosting': 'gradient_boosting_regression.joblib'
}
OUTPUT_PREFIX = 'regression_analysis_'

def run_regression_analysis():
    print("1. Loading Data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return

    # --- PREPARE DATA ---
    print("2. Preparing Data (Recreating Test Split)...")
    
    # Define Target and Features (Must match Training exactly)
    categorical_features = [
        'Scan Type', 'Wheelchair User', 'Hoist Required','Poor Mobility', 
        'Learning Difficulty', 'Cognitive Impairment', 'Interpreter Required', 
        'Sedation needed', 'Difficult IV Access', 'Hospital Transport Required', 'Contrast'
    ]
    numerical_features = ['Age']
    features = numerical_features + categorical_features
    target = 'Overrun Minutes'

    X = df[features]
    y = df[target]

    # Re-create the exact split used in training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    results = {}
    comparison_data = [] # List to store table rows
    
    # --- EVALUATE MODELS ---
    print("\n3. Evaluating Models...")
    
    for model_name, filename in MODEL_FILES.items():
        print(f"\n   --- Evaluating {model_name} ---")
        try:
            pipeline = joblib.load(filename)
        except FileNotFoundError:
            print(f"   ! Error: {filename} not found.")
            continue
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Add to table data
        comparison_data.append({
            'Model': model_name,
            'MAE (Minutes)': round(mae, 2),
            'RMSE (Minutes)': round(rmse, 2),
            'R2 Score': round(r2, 4)
        })
        
        # Extract Feature Importances
        importances = None
        feature_names = None
        
        # Try to get importances from the regressor step
        if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
            importances = pipeline.named_steps['regressor'].feature_importances_
            
            # Try to get feature names
            try:
                preprocessor = pipeline.named_steps['preprocessor']
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = [f"Feature {i}" for i in range(len(importances))]

        results[model_name] = {
            'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'y_pred': y_pred,
            'importances': importances,
            'feature_names': feature_names
        }
        
        print(f"   -> MAE: {mae:.2f} | R2: {r2:.4f}")

    if not results: return

    # --- GENERATE COMPARISON TABLE ---
    print("\n4. Generating Comparison Table...")
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate % Improvement (vs the first model in list, usually Random Forest)
    base_mae = comparison_df.iloc[0]['MAE (Minutes)']
    comparison_df['% Diff vs Baseline'] = comparison_df.apply(
        lambda x: f"{((base_mae - x['MAE (Minutes)']) / base_mae * 100):.1f}%" if x['Model'] != comparison_df.iloc[0]['Model'] else "-", 
        axis=1
    )

    print("\n" + "="*60)
    print("FINAL REGRESSION MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60 + "\n")
    
    # Save to CSV
    comparison_df.to_csv('regression_comparison_table.csv', index=False)
    print("Table saved as 'regression_comparison_table.csv'")

    # --- PLOT 1: METRICS COMPARISON ---
    print("\n5. Generating Plots...")
    
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({'Model': name, 'Metric': 'MAE (Lower is Better)', 'Score': res['MAE']})
        metrics_data.append({'Model': name, 'Metric': 'RMSE (Lower is Better)', 'Score': res['RMSE']})
        metrics_data.append({'Model': name, 'Metric': 'R2 Score (Higher is Better)', 'Score': res['R2']})
    
    metrics_df = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='viridis')
    plt.title('Regression Model Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}metrics_comparison.png')
    print("   - Saved Metrics Comparison")

    # --- PLOT 2: ACTUAL VS PREDICTED ---
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        sns.scatterplot(x=y_test, y=res['y_pred'], label=name, alpha=0.6)
    
    # Ideal line
    min_val, max_val = min(y_test.min(), 0), max(y_test.max(), 100)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Actual vs Predicted Overrun Minutes')
    plt.xlabel('Actual Overrun')
    plt.ylabel('Predicted Overrun')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}actual_vs_predicted.png')
    print("   - Saved Actual vs Predicted Plot")

    # --- PLOT 3: RESIDUAL DISTRIBUTION ---
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        residuals = y_test - res['y_pred']
        sns.kdeplot(residuals, label=f"{name} (Mean Error: {residuals.mean():.2f})", fill=True, alpha=0.3)
    
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Residual Error Distribution')
    plt.xlabel('Error (Actual - Predicted)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}residuals_distribution.png')
    print("   - Saved Residuals Plot")

    # --- PLOT 4: FEATURE IMPORTANCE ---
    feature_data = []
    for name, res in results.items():
        if res['importances'] is not None:
            # Clean names
            clean_names = [f.replace('cat__', '').replace('num__', '') for f in res['feature_names']]
            temp_df = pd.DataFrame({
                'Feature': clean_names, 
                'Importance': res['importances'], 
                'Model': name
            })
            temp_df = temp_df.sort_values(by='Importance', ascending=False).head(10)
            feature_data.append(temp_df)
    
    if feature_data:
        feat_df = pd.concat(feature_data)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', hue='Model', data=feat_df, palette='magma')
        plt.title('Top 10 Feature Importances: RFR vs GBR')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PREFIX}feature_importance.png')
        print("   - Saved Feature Importance Comparison")

    print("\nDone! All regression analysis files generated.")

if __name__ == "__main__":
    run_regression_analysis()