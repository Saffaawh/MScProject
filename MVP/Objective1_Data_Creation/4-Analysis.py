import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_FILE = 'final_historical_mri_dataset.csv'

def generate_full_flag_table():
    print("1. Loading Data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return

    # 1. Define Mappings: { CSV_Column_Name : Report_Display_Name }
    # I have matched these to the standard names in your dataset based on our history
    flag_map = {
        'Wheelchair User': 'Wheelchair User',
        'Hoist Required': 'Hoist Required',
        'Sedation needed': 'Claustrophobic (Sedation)', 
        'Poor Mobility': 'Poor Mobility',
        'Cognitive Conditions': 'Cognitive Impairment',  # Assuming CSV uses 'Conditions' or 'Impairment'
        'Learning Difficulty': 'Learning Difficulty',
        'Interpreter Required': 'Interpreter Required',
        'Difficult IV Access': 'DIVA (Difficult IV Access)',
        'Hospital Transport Required': 'Hospital Transport'
    }
    
    # Check which column actually exists for Cognitive
    if 'Cognitive Conditions' not in df.columns and 'Cognitive Impairment' in df.columns:
        # Swap key if the CSV uses the other name
        del flag_map['Cognitive Conditions']
        flag_map['Cognitive Impairment'] = 'Cognitive Impairment'

    # 2. Determine Class (if not already in CSV)
    if 'Class' not in df.columns:
        conditions = [
            (df['Overrun Minutes'] > 20),
            (df['Overrun Minutes'] > 0)
        ]
        choices = ['Class 3', 'Class 2']
        df['Class'] = np.select(conditions, choices, default='Class 1')

    # 3. Build the Data Rows
    table_data = []

    # --- A. The "No Flags (Control)" Row ---
    # Find patients where ALL mapped flags are "No"
    mask_no_flags = pd.Series([True] * len(df))
    valid_csv_cols = [c for c in flag_map.keys() if c in df.columns]
    
    for col in valid_csv_cols:
        is_present = df[col].astype(str).str.lower().isin(['yes', 'true', '1'])
        mask_no_flags = mask_no_flags & (~is_present)
    
    control_group = df[mask_no_flags]
    total_ctrl = len(control_group)
    c3_ctrl = len(control_group[control_group['Class'] == 'Class 3'])
    c2_ctrl = len(control_group[control_group['Class'] == 'Class 2'])
    
    table_data.append({
        'Delay Reason': 'No Flags (Control)',
        'Total Cases': total_ctrl,
        'Major Delay (Class 3)': c3_ctrl,
        '% Major': (c3_ctrl / total_ctrl * 100) if total_ctrl > 0 else 0,
        '% Minor (Class 2)': (c2_ctrl / total_ctrl * 100) if total_ctrl > 0 else 0
    })

    # --- B. The Specific Flags (in the order you requested) ---
    # We loop through your desired display order
    target_order = [
        'Wheelchair User', 'Hoist Required', 'Claustrophobic (Sedation)', 
        'Poor Mobility', 'Cognitive Impairment', 'Learning Difficulty', 
        'Interpreter Required', 'DIVA (Difficult IV Access)', 'Hospital Transport'
    ]
    
    # Create a reverse map to find the CSV column for the Display Name
    display_to_csv = {v: k for k, v in flag_map.items()}

    for display_name in target_order:
        csv_col = display_to_csv.get(display_name)
        
        if not csv_col or csv_col not in df.columns:
            print(f"Warning: Column for '{display_name}' not found in CSV.")
            continue
            
        subset = df[df[csv_col].astype(str).str.lower().isin(['yes', 'true', '1'])]
        
        total = len(subset)
        c3 = len(subset[subset['Class'] == 'Class 3'])
        c2 = len(subset[subset['Class'] == 'Class 2'])
        
        table_data.append({
            'Delay Reason': display_name,
            'Total Cases': total,
            'Major Delay (Class 3)': c3,
            '% Major': (c3 / total * 100) if total > 0 else 0,
            '% Minor (Class 2)': (c2 / total * 100) if total > 0 else 0
        })

    # 4. Output
    result_df = pd.DataFrame(table_data)
    
    # Format Percentages
    result_df['% Major'] = result_df['% Major'].map('{:.1f}%'.format)
    result_df['% Minor (Class 2)'] = result_df['% Minor (Class 2)'].map('{:.1f}%'.format)
    
    
    print(result_df.to_string(index=False))
    print("="*90 + "\n")

if __name__ == "__main__":
    generate_full_flag_table()