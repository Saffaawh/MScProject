import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import joblib
import warnings
import random

warnings.filterwarnings("ignore")

# --- FILES ---
REQUESTS_FILE = 'sampled_mri_requests_dataset.csv'
CLASSIFIER_MODEL = 'random_forest_model_pipeline.joblib'
REGRESSOR_MODEL = 'random_forest_regression.joblib'

# --- CONFIG ---
START_DATE = date(2025, 1, 2)
END_DATE = date(2025, 3, 31)

scan_types_details = {
    'Brain': 30, 'Brain with Contrast': 45,
    'Cervical Spine': 30, 'Cervical Spine with Contrast': 45,
    'Thoracic Spine': 30, 'Thoracic Spine with Contrast': 45,
    'Lumbar Spine': 30, 'Lumbar Spine with Contrast': 45,
    'Whole Spine': 45, 'Whole Spine with Contrast': 60
}

scanners = {
    'MRI Scanner 1': {'Type': 'Wide Bore', 'Start': 8, 'End': 20},
    'MRI Scanner 2': {'Type': 'Standard', 'Start': 8, 'End': 20},
    'MRI Scanner 3': {'Type': 'Small Bore', 'Start': 8, 'End': 20},
    'MRI Scanner 4': {'Type': 'Research', 'Start': 8, 'End': 20}
}

FLAG_COLUMNS = [
    'Hoist Required', 'Sedation needed', 'Hospital Transport Required', 
    'Wheelchair User', 'Poor Mobility', 'Cognitive Conditions', 
    'Learning Difficulty', 'Cognitive Impairment', 'Interpreter Required', 
    'Difficult IV Access'
]

def load_and_predict():
    """Loads data, overrides dates to ensure simulation works, and predicts."""
    try:
        df = pd.read_csv(REQUESTS_FILE)
    except FileNotFoundError:
        print(f"Error: {REQUESTS_FILE} not found.")
        exit()

    if 'PatientNumber' in df.columns:
        df.rename(columns={'PatientNumber': 'Patient Number'}, inplace=True)

    try:
        clf = joblib.load(CLASSIFIER_MODEL)
        reg = joblib.load(REGRESSOR_MODEL)
    except FileNotFoundError:
        print("Error: Model files not found.")
        exit()

    # --- DATE OVERRIDE ---
    # Ensures all patients are available from the start of the simulation
    print("Adjusting patient request dates to start from 2025-01-02...")
    df['Request Date'] = [START_DATE + timedelta(days=x % 5) for x in range(len(df))]

    # Normalize Binary Columns
    binary_cols = [
        'Mobility Needs', 'Wheelchair User', 'Hoist Required', 'Poor Mobility',
        'Cognitive Conditions', 'Learning Difficulty', 'Cognitive Impairment', 
        'Interpreter Required', 'Sedation needed', 'Difficult IV Access', 
        'Hospital Transport Required'
    ]
    
    for col in binary_cols:
        if col not in df.columns:
            df[col] = "No"
        else:
            df[col] = df[col].fillna("No")
            df[col] = df[col].apply(lambda x: "Yes" if str(x).lower() in ['yes', 'true', '1'] else "No")

    if 'Contrast' in df.columns:
         df['Contrast'] = df['Contrast'].apply(lambda x: "Yes" if str(x).lower() in ['yes', 'true', '1'] else "No")

    feature_cols = [
        'Age', 'Gender', 'Scan Type', 'Mobility Needs', 'Wheelchair User',
        'Hoist Required', 'Poor Mobility', 'Cognitive Conditions',
        'Learning Difficulty', 'Cognitive Impairment', 'Interpreter Required',
        'Sedation needed', 'Contrast', 'Difficult IV Access',
        'Hospital Transport Required'
    ]

    # Predict
    X_final = df[feature_cols]
    df['Predicted_Class'] = clf.predict(X_final)
    df['Predicted_Overrun'] = reg.predict(X_final)

    # Calculate Duration
    df['Duration'] = df.apply(
        lambda row: scan_types_details.get(row['Scan Type'], 30) + max(0, round(row['Predicted_Overrun'])),
        axis=1
    )

    return df

def is_weekend_check(dt):
    return dt.weekday() >= 5

def is_valid_slot(patient, scanner_name, current_dt, end_dt, scanner_stats, daily_stats, strict_mode=True):
    
    hoist = patient.get('Hoist Required') == "Yes"
    sedation = patient.get('Sedation needed') == "Yes"
    transport = patient.get('Hospital Transport Required') == "Yes"
    
    complexity_score = sum(1 for col in FLAG_COLUMNS if patient.get(col) == "Yes")
    has_flag = complexity_score > 0
    
    is_weekend = is_weekend_check(current_dt)

    # --- HARD RULES (Safety & Equipment) ---
    if is_weekend and patient['Predicted_Class'] == 'Class 3':
        return False
    
    if is_weekend and (hoist or sedation or transport):
        return False
        
    if sedation and scanner_name == 'MRI Scanner 3':
        return False
        
    if scanner_name == 'MRI Scanner 2' and hoist:
        return False
        
    if hoist and (current_dt.hour >= 17 or scanner_stats['Hoists'] >= 2):
        return False

    if transport and (current_dt.hour >= 17 or daily_stats['Transport'] >= 3):
        return False

    if sedation and current_dt.hour >= 17:
        return False
        
    if patient['Predicted_Class'] == 'Class 3' and current_dt.hour >= 17:
        return False

    if sedation and scanner_stats['Sedation_Streak'] >= 2:
        return False
        
    if scanner_name == 'MRI Scanner 4' and current_dt.weekday() == 2 and current_dt.hour >= 13:
        return False

    # --- SOFT RULES (Balancing) ---
    if strict_mode:
        # 1. No Flags at 8:00 AM
        if has_flag and current_dt.hour == 8 and current_dt.minute <= 10:
            return False

        # 2. No Flags at End of Day (20:00)
        if has_flag and end_dt.hour >= 20:
            return False

        # 3. Max 2 Consecutive Flags
        if has_flag and scanner_stats['Consecutive_Flags'] >= 2:
            return False

    return True

def run_scheduler(df):
    schedule = []
    
    df['Complexity_Score'] = df.apply(
        lambda row: sum(1 for col in FLAG_COLUMNS if row.get(col) == "Yes"), axis=1
    )
    
    # Sort: Date first, then Complexity
    df['Request Date'] = pd.to_datetime(df['Request Date']).dt.date
    df = df.sort_values(by=['Request Date', 'Complexity_Score'], ascending=[True, False])

    high_complex_list = df[df['Complexity_Score'] >= 3].to_dict('records')
    medium_complex_list = df[(df['Complexity_Score'] >= 1) & (df['Complexity_Score'] < 3)].to_dict('records')
    low_complex_list = df[df['Complexity_Score'] == 0].to_dict('records')
    
    master_pools = {
        'high': high_complex_list,
        'medium': medium_complex_list,
        'low': low_complex_list
    }
    
    current_date = START_DATE
    
    print("Starting simulation...")

    while current_date <= END_DATE and (master_pools['high'] or master_pools['medium'] or master_pools['low']):
        
        remaining = len(master_pools['high']) + len(master_pools['medium']) + len(master_pools['low'])
        print(f"Processing Day: {current_date} | Remaining: {remaining}   ", end='\r')

        scanner_stats = {s: {'Hoists': 0, 'Sedation_Streak': 0, 'Consecutive_Flags': 0} for s in scanners}
        
        # --- CRITICAL FIX: TRACK ACTUAL FINISH TIMES ---
        # Initialize all scanners as free at 08:00
        scanner_finish_times = {s: datetime.combine(current_date, time(8, 0)) for s in scanners}
        
        daily_stats = {'Transport': 0}
        
        current_time = datetime.combine(current_date, time(8, 0))
        day_end = datetime.combine(current_date, time(20, 0))
        
        # Filter pools for today
        todays_pools = {
            'high': [p for p in master_pools['high'] if p['Request Date'] <= current_date],
            'medium': [p for p in master_pools['medium'] if p['Request Date'] <= current_date],
            'low': [p for p in master_pools['low'] if p['Request Date'] <= current_date]
        }

        while current_time < day_end:
            
            # If nothing left to book today, stop checking
            if not any(todays_pools.values()):
                break

            for scanner_name in scanners:
                # --- OVERLAP PREVENTION ---
                # Strictly check if this scanner is free right now.
                # If scanner finishes at 08:30, and current_time is 08:10, SKIP IT.
                if scanner_finish_times[scanner_name] > current_time:
                    continue

                # Ensure we don't start a scan past closing time
                if current_time >= day_end:
                    continue
                
                patient_booked = False
                
                # Attempt Booking: Try Strict first, then Relaxed if stuck
                for strict in [True, False]: 
                    if patient_booked: break
                    
                    for priority in ['high', 'medium', 'low']:
                        if patient_booked: break
                        
                        active_pool = todays_pools[priority]
                        if not active_pool: continue
                            
                        best_daily_idx = None 
                        best_patient = None
                        
                        for idx, patient in enumerate(active_pool):
                            dur = int(patient['Duration'])
                            end = current_time + timedelta(minutes=dur)
                            
                            if end > day_end:
                                continue
                            
                            # Check Constraints
                            if is_valid_slot(patient, scanner_name, current_time, end, 
                                           scanner_stats[scanner_name], daily_stats, strict_mode=strict):
                                best_patient = patient
                                best_daily_idx = idx
                                break
                        
                        if best_patient:
                            # --- BOOKING ---
                            flags = []
                            for col in FLAG_COLUMNS:
                                if col in best_patient and best_patient[col] == "Yes":
                                    clean_name = col.replace(' Required', '').replace(' needed', '').replace(' User', '')
                                    flags.append(clean_name)
                            
                            has_flag = len(flags) > 0

                            schedule.append({
                                'Scanner': scanner_name,
                                'Date': current_date.strftime('%Y-%m-%d'),
                                'Start Time': current_time.strftime('%H:%M'),
                                'End Time': end.strftime('%H:%M'),
                                'Patient Number': best_patient['Patient Number'],
                                'Scan Type': best_patient['Scan Type'],
                                'Flags': ", ".join(flags) if flags else "None",
                                'Predicted_Overrun': round(best_patient['Predicted_Overrun'], 1),
                                'Predicted_Class': best_patient['Predicted_Class'],
                                'Duration': dur,
                                'Complexity': priority
                            })
                            
                            # Update Stats
                            if best_patient.get('Hoist Required') == "Yes":
                                scanner_stats[scanner_name]['Hoists'] += 1
                            if best_patient.get('Sedation needed') == "Yes":
                                scanner_stats[scanner_name]['Sedation_Streak'] += 1
                            else:
                                scanner_stats[scanner_name]['Sedation_Streak'] = 0
                            if best_patient.get('Hospital Transport Required') == "Yes":
                                daily_stats['Transport'] += 1

                            if has_flag:
                                scanner_stats[scanner_name]['Consecutive_Flags'] += 1
                            else:
                                scanner_stats[scanner_name]['Consecutive_Flags'] = 0
                            
                            # Remove Patient
                            todays_pools[priority].pop(best_daily_idx)
                            p_num = best_patient['Patient Number']
                            master_pools[priority] = [p for p in master_pools[priority] if p['Patient Number'] != p_num]

                            # --- UPDATE SCANNER FINISH TIME ---
                            # This locks the scanner until the patient is done + 5 mins gap
                            scanner_finish_times[scanner_name] = end + timedelta(minutes=5)
                            
                            patient_booked = True
            
            # Advance Global Clock
            current_time += timedelta(minutes=5)
        
        current_date += timedelta(days=1)

    print("\nSimulation Complete.")
    return schedule

def generate_html_visualisation(df_sched, filename):
    if df_sched.empty:
        print(f"Skipping HTML generation: No appointments found in {filename}")
        return

    colors = {
        'Hoist': '#ff4d4d', 'Transport': '#ff9933', 'Wheelchair': '#ffff66',
        'Sedation': '#cc99ff', 'Interpreter': '#66b3ff', 'None': '#99ff99',
        'Standard': '#99ff99'
    }

    if 'Date' not in df_sched.columns:
        print("Error: 'Date' column missing.")
        return

    df_sched['Day'] = pd.to_datetime(df_sched['Date']).dt.day_name()

    html = """<html><head><style>
            body { font-family: Arial, sans-serif; font-size: 12px; }
            h1 { text-align: center; }
            .day-container { border: 1px solid #ccc; margin: 20px; padding: 10px; background: #fff; }
            .scanner-row { display: flex; align-items: center; margin-bottom: 5px; border-bottom: 1px solid #eee; }
            .scanner-label { width: 100px; font-weight: bold; }
            .timeline { display: flex; flex-grow: 1; position: relative; height: 40px; background-color: #f0f0f0; border-radius: 4px; }
            .slot { position: absolute; height: 30px; top: 5px; border-radius: 4px; font-size: 10px;
                    overflow: hidden; white-space: nowrap; text-align: center; line-height: 30px;
                    border: 1px solid #666; color: #000; cursor: help; }
            .slot:hover { opacity: 0.8; border: 2px solid #000; z-index: 10; font-weight: bold;}
            .legend { text-align: center; margin-bottom: 20px; }
            .legend span { margin-right: 15px; padding: 5px 10px; border: 1px solid #ccc; }
        </style></head><body><h1>MRI Schedule Visualisation (Balanced ML)</h1><div class="legend">"""

    for label, color in colors.items():
        html += f'<span style="background-color: {color}">{label}</span>'
    html += '</div>'

    dates = sorted(df_sched['Date'].unique())[:7]

    for date_str in dates:
        day_df = df_sched[df_sched['Date'] == date_str]
        day_name = day_df.iloc[0]['Day']

        html += f'<div class="day-container"><h3>{day_name} ({date_str})</h3>'

        for scanner in scanners:
            html += f'<div class="scanner-row"><div class="scanner-label">{scanner}</div><div class="timeline">'
            scan_data = day_df[day_df['Scanner'] == scanner]

            for _, row in scan_data.iterrows():
                start_h, start_m = map(int, row['Start Time'].split(':'))
                start_min = (start_h - 8) * 60 + start_m

                first_flag = row['Flags'].split(",")[0] if row['Flags'] != "None" else "None"
                bg = colors.get(first_flag, "#e0e0e0")
                label = first_flag if first_flag != "None" else "Std"

                tooltip = (
                    f"Patient: {row['Patient Number']}\n"
                    f"Scan: {row['Scan Type']}\n"
                    f"Flags: {row['Flags']}\n"
                    f"Risk Class: {row['Predicted_Class']}\n"
                    f"Predicted Overrun: +{row['Predicted_Overrun']}m\n"
                    f"Duration: {row['Duration']}m\n"
                    f"Complexity: {row['Complexity']}\n"
                    f"Time: {row['Start Time']} - {row['End Time']}"
                )

                html += (
                    f'<div class="slot" '
                    f'style="left:{start_min*1.5}px; width:{row["Duration"]*1.5}px; background-color:{bg};" '
                    f'title="{tooltip}">{label}</div>'
                )

            html += '</div></div>' 

        html += '</div>' 

    html += "</body></html>"

    with open(filename, "w") as f:
        f.write(html)

    print(f"HTML Generated: {filename}")

if __name__ == "__main__":
    df_new = load_and_predict()
    print("Total patients loaded:", len(df_new))
    
    schedule = run_scheduler(df_new)
    
    if not schedule:
        print("\n[!] CRITICAL: No appointments were booked.")
    else:
        print(f"\nSuccess! Appointments booked: {len(schedule)}")
        df_sched = pd.DataFrame(schedule)
        df_sched.to_csv("final_ml_schedule.csv", index=False)
        generate_html_visualisation(df_sched, "mri_schedule_visualisation_ml.html")
        print("Scheduling complete.")