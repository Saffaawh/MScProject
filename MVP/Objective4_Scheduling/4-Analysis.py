import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix

# --- CONFIGURATION ---
OUTPUT_PREFIX = 'comparison_results_'

# The "Standard of Care" Rule Book (Baseline)
SCAN_RULES = {
    'Brain': 30, 'Brain with Contrast': 45,
    'Cervical Spine': 30, 'Cervical Spine with Contrast': 45,
    'Thoracic Spine': 30, 'Thoracic Spine with Contrast': 45,
    'Lumbar Spine': 30, 'Lumbar Spine with Contrast': 45,
    'Whole Spine': 45, 'Whole Spine with Contrast': 60
}

def run_comparison():
    print("Loading datasets...")
    
    # 1. Load the datasets
    try:
        df_ml = pd.read_csv('final_ml_schedule.csv')
        df_heuristic = pd.read_csv('final_heuristic_schedule_averages.csv')
        df_history = pd.read_csv('final_historical_mri_dataset.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Standardize Column Names
    # ML File
    if 'Predicted_Class' in df_ml.columns:
        df_ml.rename(columns={'Predicted_Class': 'Class_ML'}, inplace=True)
    elif 'Risk_Class' in df_ml.columns:
        df_ml.rename(columns={'Risk_Class': 'Class_ML'}, inplace=True)
    
    if 'Patient Number' in df_ml.columns:
        df_ml.rename(columns={'Patient Number': 'PatientID'}, inplace=True)
        
    df_ml.rename(columns={'Duration': 'Duration_ML'}, inplace=True)

    # Heuristic File
    df_heuristic = df_heuristic.rename(columns={
        'Patient Number': 'PatientID', 
        'Duration': 'Duration_Heuristic',
        'Predicted_Class': 'Class_Heuristic'
    })
    
    # History File
    df_history = df_history.rename(columns={
        'PatientNumber': 'PatientID', 
        'Actual Scan Duration': 'Duration_Actual', 
        'Class': 'Class_Actual'
    })

    # 3. GENERATE BASELINE (Standard Rules)
    scan_col = 'Scan Type' 
    if scan_col not in df_history.columns:
        print(f"WARNING: Could not find '{scan_col}' in history file.")
        return

    df_history['Duration_Baseline'] = df_history[scan_col].map(SCAN_RULES).fillna(30)

    # 4. Merge Everything
    print("Merging data...")
    # Merge ML + History
    merged = pd.merge(df_ml[['PatientID', 'Duration_ML', 'Class_ML']], 
                      df_history[['PatientID', 'Duration_Actual', 'Duration_Baseline', 'Class_Actual']], 
                      on='PatientID', how='inner')

    # Merge Heuristic
    merged = pd.merge(merged, 
                      df_heuristic[['PatientID', 'Duration_Heuristic', 'Class_Heuristic']], 
                      on='PatientID', how='inner')

    print(f"Successfully matched {len(merged)} patients.")

    if len(merged) == 0: 
        return

    # --- PART 1: REGRESSION ANALYSIS ---
    merged['Error_Baseline'] = (merged['Duration_Baseline'] - merged['Duration_Actual']).abs()
    merged['Error_Heuristic'] = (merged['Duration_Heuristic'] - merged['Duration_Actual']).abs()
    merged['Error_ML'] = (merged['Duration_ML'] - merged['Duration_Actual']).abs()

    mae_base = merged['Error_Baseline'].mean()
    mae_heur = merged['Error_Heuristic'].mean()
    mae_ml = merged['Error_ML'].mean()
    
    rmse_base = np.sqrt(mean_squared_error(merged['Duration_Actual'], merged['Duration_Baseline']))
    rmse_heur = np.sqrt(mean_squared_error(merged['Duration_Actual'], merged['Duration_Heuristic']))
    rmse_ml = np.sqrt(mean_squared_error(merged['Duration_Actual'], merged['Duration_ML']))

    print("\n" + "="*80)
    print(f"PART 1: REGRESSION EFFICIENCY (N={len(merged)})")
    print("="*80)
    print(f"{'Metric':<25} | {'Baseline (Rules)':<18} | {'Heuristic':<15} | {'ML Model':<15}")
    print("-" * 80)
    print(f"{'Mean Error (MAE)':<25} | {mae_base:<18.2f} | {mae_heur:<15.2f} | {mae_ml:<15.2f}")
    print(f"{'RMSE':<25} | {rmse_base:<18.2f} | {rmse_heur:<15.2f} | {rmse_ml:<15.2f}")
    print("-" * 80)

    # --- PART 2: OPERATIONAL IMPACT ---
    # Baseline Overtime
    diff_base = merged['Duration_Baseline'] - merged['Duration_Actual']
    ot_base = diff_base[diff_base < 0].abs().sum()
    idle_base = diff_base[diff_base > 0].sum()

    # Heuristic Overtime
    diff_heur = merged['Duration_Heuristic'] - merged['Duration_Actual']
    ot_heur = diff_heur[diff_heur < 0].abs().sum()
    idle_heur = diff_heur[diff_heur > 0].sum()

    # ML Overtime
    diff_ml = merged['Duration_ML'] - merged['Duration_Actual']
    ot_ml = diff_ml[diff_ml < 0].abs().sum()
    idle_ml = diff_ml[diff_ml > 0].sum()

    print("\n" + "="*80)
    print("PART 2: OPERATIONAL IMPACT (BUSINESS CASE)")
    print("="*80)
    print(f"{'Metric':<30} | {'Baseline':<15} | {'Heuristic':<15} | {'ML Model':<15}")
    print("-" * 80)
    print(f"{'Total Unplanned Overtime (m)':<30} | {ot_base:<15.0f} | {ot_heur:<15.0f} | {ot_ml:<15.0f}")
    print(f"{'Total Wasted Idle Time (m)':<30} | {idle_base:<15.0f} | {idle_heur:<15.0f} | {idle_ml:<15.0f}")
    print("-" * 80)

    # Calculate total inefficiency for stacked chart
    total_base = ot_base + idle_base
    total_heur = ot_heur + idle_heur
    total_ml = ot_ml + idle_ml

    # --- PART 3: CLASSIFICATION ANALYSIS ---
    # Ensure classes are strings and stripped
    for col in ['Class_Actual', 'Class_Heuristic', 'Class_ML']:
        merged[col] = merged[col].astype(str).str.strip()
        
    labels = ['Class 1', 'Class 2', 'Class 3']
    
    print("\n" + "="*80)
    print("PART 3: CLASSIFICATION PERFORMANCE")
    print("="*80)
    
    print("\n--- HEURISTIC CLASSIFICATION REPORT ---")
    print(classification_report(merged['Class_Actual'], merged['Class_Heuristic'], labels=labels, zero_division=0))
    
    print("\n--- ML MODEL CLASSIFICATION REPORT ---")
    print(classification_report(merged['Class_Actual'], merged['Class_ML'], labels=labels, zero_division=0))

    # --- GENERATE VISUALS ---
    print("\nGenerating Visuals...")

    # Plot 1: MAE Comparison
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MAE', 'MAE', 'RMSE', 'RMSE', 'RMSE'],
        'Score': [mae_base, mae_heur, mae_ml, rmse_base, rmse_heur, rmse_ml],
        'Model': ['Baseline', 'Heuristic', 'ML', 'Baseline', 'Heuristic', 'ML']
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='viridis')
    plt.title('Prediction Accuracy: Baseline vs Heuristic vs ML')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f'{OUTPUT_PREFIX}regression_metrics_3way.png')
    print(" - Saved regression_metrics_3way.png")

    # Plot 2: STACKED BAR CHART for Business Impact
    plt.figure(figsize=(12, 7))

    # Create data for stacked bars
    models = ['Baseline', 'Heuristic', 'ML Model']
    overtime = [ot_base, ot_heur, ot_ml]
    idle_time = [idle_base, idle_heur, idle_ml]
    total_inefficiency = [total_base, total_heur, total_ml]

    # Create the stacked bar chart
    bars1 = plt.bar(models, overtime, color='#ff6b6b', label='Overtime (Delays)', 
                   edgecolor='black', linewidth=1.2)
    bars2 = plt.bar(models, idle_time, bottom=overtime, color='#4ecdc4', 
                   label='Idle Time (Wasted)', edgecolor='black', linewidth=1.2)

    # Add value labels on top of each stack
    for i, (ot, idle, total) in enumerate(zip(overtime, idle_time, total_inefficiency)):
        # Label for overtime section
        if ot > 0:
            plt.text(i, ot/2, f'{ot:,.0f}m', ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='white')
        
        # Label for idle time section
        if idle > 0:
            plt.text(i, ot + idle/2, f'{idle:,.0f}m', ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='white')
        
        # Total label at top of bar
        plt.text(i, total + (max(total_inefficiency) * 0.02), f'Total: {total:,.0f}m', 
                ha='center', va='bottom', fontweight='bold', fontsize=11, color='#333333')

   
  
  

   
   
   

    # Customize the plot
    plt.title('Operational Impact Comparison: Balancing Overtime vs Idle Time', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Scheduling Approach', fontsize=12, fontweight='bold')
    plt.ylabel('Time (minutes)', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust y-limit for better visualization
    max_total = max(total_inefficiency)
    plt.ylim(0, max_total * 1.15)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}efficiency_impact_stacked.png', dpi=300, bbox_inches='tight')
    print(" - Saved efficiency_impact_stacked.png")

    # Plot 3: Side-by-Side Confusion Matrices
    cm_heur = confusion_matrix(merged['Class_Actual'], merged['Class_Heuristic'], labels=labels)
    cm_ml = confusion_matrix(merged['Class_Actual'], merged['Class_ML'], labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heuristic Heatmap
    sns.heatmap(cm_heur, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Heuristic (Rule-Based) Confusion Matrix')
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('Actual Class')

    # ML Heatmap
    sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('ML Model Confusion Matrix')
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('Actual Class')

    plt.suptitle('Risk Classification Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(" - Saved confusion_matrices.png")

    # Plot 4: Residual Plot
    merged['Resid_Base'] = merged['Duration_Baseline'] - merged['Duration_Actual']
    merged['Resid_Heur'] = merged['Duration_Heuristic'] - merged['Duration_Actual']
    merged['Resid_ML'] = merged['Duration_ML'] - merged['Duration_Actual']

    plt.figure(figsize=(12, 6))
    plt.scatter(merged['Duration_Actual'], merged['Resid_Base'], alpha=0.3, color='gray', label='Baseline', s=20)
    plt.scatter(merged['Duration_Actual'], merged['Resid_Heur'], alpha=0.3, color='red', label='Heuristic', s=20)
    plt.scatter(merged['Duration_Actual'], merged['Resid_ML'], alpha=0.4, color='blue', label='ML Model', s=20)
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.title('Residual Plot: Evolution of Accuracy')
    plt.xlabel('Actual Duration (mins)')
    plt.ylabel('Error (Predicted - Actual)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{OUTPUT_PREFIX}residual_plot_3way.png', dpi=300, bbox_inches='tight')
    print(" - Saved residual_plot_3way.png")

    print("\nDone! All visuals created.")

    # --- EXTRA: Create grouped bar chart alternative ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Overtime comparison
    bars_ot = ax1.bar(models, overtime, color=['#ff6b6b', '#ff8e8e', '#ffb1b1'], edgecolor='black', linewidth=1.2)
    ax1.set_title('Unplanned Overtime (Delays)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Minutes', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, bar in enumerate(bars_ot):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (max(overtime) * 0.02),
                f'{height:,.0f}m', ha='center', va='bottom', fontweight='bold')

    # Right: Idle Time comparison
    bars_idle = ax2.bar(models, idle_time, color=['#4ecdc4', '#6ed7d0', '#8ee1dc'], edgecolor='black', linewidth=1.2)
    ax2.set_title('Wasted Idle Time', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Minutes', fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for i, bar in enumerate(bars_idle):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (max(idle_time) * 0.02),
                f'{height:,.0f}m', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Operational Efficiency Breakdown: ML Achieves Optimal Balance', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}efficiency_impact_grouped.png', dpi=300, bbox_inches='tight')
    print(" - Saved efficiency_impact_grouped.png")

if __name__ == "__main__":
    run_comparison()