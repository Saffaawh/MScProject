import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION ---
DATA_FILE = 'final_historical_mri_dataset.csv'

# Using the saved pipelines
MODEL_FILES = {
    'Decision Tree': 'decision_tree_model_pipeline.joblib',
    'Random Forest': 'random_forest_model_pipeline.joblib' 
}

OUTPUT_PREFIX = 'comparison_analysis_'

def run_comparative_analysis():
    print("1. Loading Historical Data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return

    # --- PREPARE RAW DATA ---
    print("2. Preparing Data...")
    
    target_col = 'Class'
    # Dropping non-feature columns
    drop_cols = ['PatientNumber', 'Scheduled Date and Time', 'Actual End Time', 
                 'Actual Scan Duration', 'Overrun Minutes', target_col]
    
    # X_raw contains categorical text; the pipeline handles encoding.
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]

    # Storage for results and metrics
    results = {}
    detailed_metrics = [] # List to store row-by-row metrics for CSV
    
    # --- MODEL EVALUATION LOOP ---
    print("\n3. Evaluating Models...")
    
    for model_name, filename in MODEL_FILES.items():
        print(f"\n   --- Evaluating {model_name} ---")
        try:
            pipeline = joblib.load(filename)
        except FileNotFoundError:
            print(f"   ! Error: Model file '{filename}' not found.")
            continue
        
        # Predict
        try:
            y_pred = pipeline.predict(X_raw)
        except ValueError as e:
            print(f"   ! Prediction Error: {e}")
            continue
        
        # Metrics
        acc = accuracy_score(y, y_pred)
        # Returns a dictionary with all metrics for all classes
        report_dict = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        # --- Extract Detailed Metrics for Comparison ---
        # We iterate through the report dictionary to extract Class 1, 2, 3, macro avg, etc.
        for category, metrics in report_dict.items():
            # Skip 'accuracy' key here as it is a single float, not a dict in the report
            if category == 'accuracy':
                continue
                
            row = {
                'Model': model_name,
                'Class_or_Avg': category,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            }
            detailed_metrics.append(row)
        
        # Add a separate row for overall accuracy to the list
        detailed_metrics.append({
            'Model': model_name,
            'Class_or_Avg': 'Overall Accuracy',
            'Precision': np.nan, 
            'Recall': np.nan, 
            'F1-Score': acc, # storing accuracy in f1 column for consolidation, or keep separate
            'Support': len(y)
        })

        # Feature Importances extraction (same as before)
        importances = None
        feature_names = None
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            try:
                preprocessor = pipeline.named_steps['preprocessor']
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = [f"Feature {i}" for i in range(len(importances))]

        results[model_name] = {
            'accuracy': acc,
            'report': report_dict,
            'cm': cm,
            'importances': importances,
            'feature_names': feature_names
        }
        
        print(f"   -> Accuracy: {acc:.4f}")

    if not results:
        print("\nNo models evaluated. Exiting.")
        return

    # --- SAVE FULL METRICS REPORT ---
    print("\n4. Generating Detailed Metrics Report...")
    metrics_df = pd.DataFrame(detailed_metrics)
    
    # Reorder columns for readability
    metrics_df = metrics_df[['Model', 'Class_or_Avg', 'Precision', 'Recall', 'F1-Score', 'Support']]
    
    # Save to CSV
    metrics_csv_name = f'{OUTPUT_PREFIX}detailed_metrics.csv'
    metrics_df.to_csv(metrics_csv_name, index=False)
    
    print(f"   - Saved detailed report to {metrics_csv_name}")
    print("\n   --- METRICS PREVIEW (Top Rows) ---")
    print(metrics_df.head(10)) # Print first few rows to console

    # --- VISUALIZATION 1: CONFUSION MATRIX ---
    print("\n5. Generating Comparison Plots...")
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: axes = [axes]
    
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}\nAccuracy: {res["accuracy"]:.2%}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}confusion_matrices.png')
    print("   - Saved Confusion Matrix Comparison")

    # --- VISUALIZATION 2: METRIC COMPARISON BAR CHART (Updated) ---
    # We will use the metrics_df we created above for a cleaner plot
    # Filter for 'weighted avg' to compare overall performance
    plot_df = metrics_df[metrics_df['Class_or_Avg'].isin(['weighted avg', 'Class 3'])]
    
    # Transform for plotting: We want Model on X, Score on Y, Hue by Metric type
    # But for simplicity, let's stick to the previous bar chart style but automated
    
    plot_data = []
    for name, res in results.items():
        plot_data.append({'Model': name, 'Metric': 'Accuracy', 'Score': res['accuracy']})
        plot_data.append({'Model': name, 'Metric': 'Weighted F1', 
                          'Score': res['report']['weighted avg']['f1-score']})
        # Explicitly tracking Class 3 Recall (Safety Metric)
        c3_recall = res['report'].get('Class 3', {}).get('recall', 0)
        plot_data.append({'Model': name, 'Metric': 'Class 3 Recall (Safety)', 'Score': c3_recall})

    plot_df_clean = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=plot_df_clean, palette='viridis')
    plt.title('Model Performance: Accuracy vs Safety (Class 3 Recall)')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}metrics_bar_chart.png')
    print("   - Saved Metrics Comparison Bar Chart")

    # --- VISUALIZATION 3: FEATURE IMPORTANCE ---
    feature_data = []
    for name, res in results.items():
        if res['importances'] is not None and res['feature_names'] is not None:
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
        plt.title('Top 10 Feature Importances Comparison')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_PREFIX}feature_importance_comparison.png')
        print("   - Saved Feature Importance Comparison")

    print("\nDone! All analysis files generated.")

if __name__ == "__main__":
    run_comparative_analysis()