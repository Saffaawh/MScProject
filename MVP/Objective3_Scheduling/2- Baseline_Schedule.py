import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 1. RESTORE MISSING AVERAGES FILE ---
# This ensures the heuristic map exists
avg_data = {
    'Flag': [
        'Hoist Required', 'Hospital Transport Required', 'Sedation needed',
        'Claustrophobic',
        'Wheelchair User', 'Poor Mobility', 'Interpreter Required',
        'Learning Difficulty', 'Cognitive Impairment', 'Difficult IV Access',
        'Contrast'
    ],
    'Average Predicted Overrun Minutes': [
        37.83, 37.84, 34.54, 
        34.54, 
        22.24, 20.53, 20.43,
        19.35, 18.32, 16.28,
        8.41
    ]
}
df_avgs = pd.DataFrame(avg_data)
df_avgs.to_csv('Average_predicted_Overruns_by_flag.csv', index=False)
print("Restored 'Average_predicted_Overruns_by_flag.csv'.")

# --- 2. CONFIGURATION ---

REQUESTS_FILE = 'sampled_mri_requests_dataset.csv'
AVERAGES_FILE = 'Average_predicted_Overruns_by_flag.csv'
OUTPUT_CSV = 'final_heuristic_schedule_averages.csv'
OUTPUT_HTML = 'mri_schedule_visualisation.html'

START_DATE = date(2025, 1, 2)
END_DATE = date(2025, 3, 31)

# --- SAFETY LIMITS ---
MAX_HOISTS_PER_DAY = 2             # Manual Handling Safety Limit
MAX_CLAUSTRO_SEDATION_PER_DAY = 2  # Resource/Staff Stress Limit

# --- END OF DAY BUFFER ---
LAST_COMPLEX_START_HOUR = 17 
LAST_COMPLEX_START_MINUTE = 30

scan_types_details = {
    'Brain': 30, 'Brain with Contrast': 45,
    'Cervical Spine': 30, 'Cervical Spine with Contrast': 45,
    'Thoracic Spine': 30, 'Thoracic Spine with Contrast': 45,
    'Lumbar Spine': 30, 'Lumbar Spine with Contrast': 45,
    'Whole Spine': 45, 'Whole Spine with Contrast': 60
}

scanners = {
    'MRI Scanner 1': {'Type': 'Wide Bore', 'Start': 8, 'End': 20},
    'MRI Scanner 2': {'Type': 'Standard',  'Start': 8, 'End': 20},
    'MRI Scanner 3': {'Type': 'Small Bore', 'Start': 8, 'End': 20},
    'MRI Scanner 4': {'Type': 'Research',   'Start': 8, 'End': 20}
}

# The columns to check in the input file
FLAG_COLUMNS = [
    'Hoist Required', 'Sedation needed', 'Hospital Transport Required', 
    'Wheelchair User', 'Poor Mobility', 'Cognitive Conditions', 
    'Learning Difficulty', 'Cognitive Impairment', 'Interpreter Required', 
    'Difficult IV Access'
]

# --- 3. DATA PREPARATION ---

def load_and_prep_data():
    if not os.path.exists(REQUESTS_FILE):
        logger.error(f"File not found: {REQUESTS_FILE}")
        return pd.DataFrame()

    df_patients = pd.read_csv(REQUESTS_FILE)

    # --- FIX: Standardize Patient ID ---
    if 'PatientNumber' in df_patients.columns:
        df_patients.rename(columns={'PatientNumber': 'Patient Number'}, inplace=True)

    # --- FIX: Construct 'Delay Reason' from individual flags ---
    # The heuristic logic relies on a string description, so we build it here.
    def construct_reason(row):
        reasons = []
        for col in FLAG_COLUMNS:
            if col in row and row[col] == "Yes":
                # Clean up the name (e.g., 'Hoist Required' -> 'Hoist required')
                # We keep the wording close to the Averages file keys
                reasons.append(col) 
        
        if not reasons:
            return "None"
        return ", ".join(reasons)

    df_patients['Delay Reason'] = df_patients.apply(construct_reason, axis=1)
    
    # Load Averages Map
    df_avg = pd.read_csv(AVERAGES_FILE)
    avg_map = dict(zip(df_avg['Flag'].str.lower(), df_avg['Average Predicted Overrun Minutes']))

    # Calculation Function
    def get_extra_time(flag_str, scan_type):
        flag_str = str(flag_str).lower()
        extra = 0
        # Check substrings because 'flag_str' might be "Hoist Required, Wheelchair User"
        if 'hoist' in flag_str: extra = max(extra, avg_map.get('hoist required', 38))
        if 'transport' in flag_str: extra = max(extra, avg_map.get('hospital transport required', 38))
        if 'wheelchair' in flag_str: extra = max(extra, avg_map.get('wheelchair user', 22))
        if 'mobility' in flag_str: extra = max(extra, avg_map.get('poor mobility', 20))
        if 'sedation' in flag_str or 'claustro' in flag_str: 
            extra = max(extra, avg_map.get('sedation needed', 34))
        if 'interpreter' in flag_str: extra = max(extra, avg_map.get('interpreter required', 20))
        if 'cognitive' in flag_str: extra = max(extra, avg_map.get('cognitive impairment', 18))
        if 'learning' in flag_str: extra = max(extra, avg_map.get('learning difficulty', 19))
        if 'diva' in flag_str or 'difficult iv' in flag_str: extra = max(extra, avg_map.get('difficult iv access', 16))
        
        # Contrast adds a flat buffer
        if 'contrast' in scan_type.lower():
             extra += avg_map.get('contrast', 8)
        return round(extra)

    predicted_durations = []
    for _, row in df_patients.iterrows():
        base = scan_types_details.get(row['Scan Type'], 30)
        overrun = get_extra_time(row['Delay Reason'], row['Scan Type'])
        predicted_durations.append(base + overrun)
        
    df_patients['Predicted_Duration'] = predicted_durations
    
    # Priority Sorting (Complex patients first)
    df_patients['Is_Priority'] = df_patients['Delay Reason'].astype(str).str.contains(
        'Hospital|Hoist|Claustrophobic|Sedation', case=False, regex=True
    )
    df_patients.sort_values(by=['Is_Priority', 'Request Date'], ascending=[False, True], inplace=True)
    
    return df_patients

# --- 4. CONSTRAINT CHECKS ---

def is_valid_slot(patient, scanner_name, current_dt, end_dt):
    flag = str(patient['Delay Reason'])
    is_weekend = current_dt.weekday() >= 5
    start_hour = current_dt.hour
    start_minute = current_dt.minute
    
    # Identify Complex Patients
    is_complex = any(x in flag for x in ['Hoist', 'Hospital', 'Sedation', 'Claustro', 'Wheelchair', 'Cognitive', 'Learning', 'Mobility'])
    
    # --- RULE: NO COMPLEX AT END OF DAY ---
    if is_complex:
        if start_hour > LAST_COMPLEX_START_HOUR:
            return False
        if start_hour == LAST_COMPLEX_START_HOUR and start_minute > LAST_COMPLEX_START_MINUTE:
            return False

    # Standard Rules
    class_3_check = ['Hoist', 'Hospital', 'Sedation', 'Claustro']
    if is_weekend and any(x in flag for x in class_3_check): return False
        
    if 'Hoist' in flag and 12 <= start_hour < 14: return False
            
    if 'Hospital' in flag:
        if start_hour < 9: return False
        if end_dt.hour >= 17 and end_dt.minute > 0: return False
        
    if 'Sedation' in flag or 'Claustro' in flag:
        if scanner_name == 'MRI Scanner 3': return False 
        
    if scanner_name == 'MRI Scanner 2':
        if any(x in flag for x in ['Wheelchair', 'Hoist', 'Poor mobility']): return False
            
    if scanner_name == 'MRI Scanner 4':
        if 'Hospital' in flag: return False
        # Wednesday Closure Rule
        if current_dt.weekday() == 2 and start_hour >= 13: return False
            
    return True

def apply_scanner_closures(scanner_name, current_dt):
    """
    Enforces special closure rules, e.g. Scanner 4 closed Wed 13:00-17:00.
    """
    if scanner_name == 'MRI Scanner 4' and current_dt.weekday() == 2:
        if 13 <= current_dt.hour < 17:
            return current_dt.replace(hour=17, minute=0, second=0, microsecond=0)
    return current_dt

# --- 5. SCHEDULER ENGINE ---

def run_scheduler(df_patients):
    schedule_log = []
    queue = df_patients.to_dict('records')

    current_date = START_DATE

    print(f"Scheduling {len(queue)} patients...")

    while queue and current_date <= END_DATE:

        # Per-scanner daily limits
        scanner_stats = {
            s: {'Hoists': 0, 'Sedation_Streak': 0}
            for s in scanners
        }

        # Per-scanner availability
        scanner_avail = {
            s: datetime.combine(current_date, time(8, 0))
            for s in scanners
        }

        day_end = datetime.combine(current_date, time(20, 0))

        for scanner_name in scanners:
            curr = scanner_avail[scanner_name]
            curr = apply_scanner_closures(scanner_name, curr)

            while curr < day_end and queue:
                curr = apply_scanner_closures(scanner_name, curr)
                if curr >= day_end:
                    break

                booked_idx = None

                # 🔁 LOOP THROUGH ALL PATIENTS
                for i, p in enumerate(queue):
                    flag = str(p['Delay Reason'])
                    dur = int(p['Predicted_Duration'])
                    end = curr + timedelta(minutes=dur)

                    if end > day_end:
                        continue

                    # --- DAILY LIMITS ---
                    if 'Hoist' in flag and scanner_stats[scanner_name]['Hoists'] >= MAX_HOISTS_PER_DAY:
                        continue
                    if ('Sedation' in flag or 'Claustro' in flag) and scanner_stats[scanner_name]['Sedation_Streak'] >= MAX_CLAUSTRO_SEDATION_PER_DAY:
                        continue

                    # --- VALIDITY CHECK ---
                    if not is_valid_slot(p, scanner_name, curr, end):
                        continue

                    # Fit found
                    booked_idx = i
                    break

                if booked_idx is None:
                    curr += timedelta(minutes=10)
                    continue

                # Book the patient
                p = queue.pop(booked_idx)
                
                # Flag to save (Composite string)
                flag_to_save = str(p['Delay Reason'])

                schedule_log.append({
                    'Scanner': scanner_name,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Day': current_date.strftime('%A'),
                    'Start Time': curr.strftime('%H:%M'),
                    'End Time': end.strftime('%H:%M'),
                    'Patient Number': p['Patient Number'],
                    'Age': p['Age'],
                    'Clinical Indication': p.get('Clinical Indication', 'Unknown'),
                    'Delay Reason': flag_to_save,
                    'Scan Type': p['Scan Type'],
                    'Duration': dur
                })

                # Update scanner stats
                if 'Hoist' in flag_to_save:
                    scanner_stats[scanner_name]['Hoists'] += 1

                if 'Sedation' in flag_to_save or 'Claustro' in flag_to_save:
                    scanner_stats[scanner_name]['Sedation_Streak'] += 1
                else:
                    scanner_stats[scanner_name]['Sedation_Streak'] = 0

                curr = end + timedelta(minutes=5)

            scanner_avail[scanner_name] = curr

        current_date += timedelta(days=1)

    return schedule_log

# --- 6. VISUALISATION ---

def generate_html_visualisation(df_sched, filename):
    colors = {
        'Hoist': '#ff4d4d', 'Transport': '#ff9933', 'Wheelchair': '#ffff66',
        'Sedation': '#cc99ff', 'Claustro': '#cc99ff', 'Interpreter': '#66b3ff', 'None': '#99ff99'
    }
    
    html = """<html><head><style>
            body { font-family: Arial, sans-serif; font-size: 12px; }
            h1 { text-align: center; }
            .day-container { border: 1px solid #ccc; margin: 20px; padding: 10px; background: #fff; }
            .scanner-row { display: flex; align-items: center; margin-bottom: 5px; border-bottom: 1px solid #eee; }
            .scanner-label { width: 100px; font-weight: bold; }
            .timeline { display: flex; flex-grow: 1; position: relative; height: 40px; background-color: #f0f0f0; border-radius: 4px; }
            .slot { position: absolute; height: 30px; top: 5px; border-radius: 4px; font-size: 10px; overflow: hidden; white-space: nowrap; text-align: center; line-height: 30px; border: 1px solid #666; color: #000; cursor: help; }
            .slot:hover { opacity: 0.8; border: 2px solid #000; z-index: 10; font-weight: bold;}
            .legend { text-align: center; margin-bottom: 20px; }
            .legend span { margin-right: 15px; padding: 5px 10px; border: 1px solid #ccc; }
        </style></head><body><h1>MRI Schedule Visualisation</h1><div class="legend">"""
    
    seen = []
    for r, c in colors.items():
        if c not in seen:
            lbl = "Sedation / Claustrophobia" if "Sedation" in r else r
            html += f'<span style="background-color: {c}">{lbl}</span>'
            seen.append(c)
    html += '<span style="background-color: #e0e0e0">Other</span></div>'
    
    dates = sorted(df_sched['Date'].unique())[:7]
    for date_str in dates:
        day_df = df_sched[df_sched['Date'] == date_str]
        html += f'<div class="day-container"><h3>{day_df.iloc[0]["Day"]} ({date_str})</h3>'
        for scanner in scanners:
            html += f'<div class="scanner-row"><div class="scanner-label">{scanner}</div><div class="timeline">'
            scan_data = day_df[day_df['Scanner'] == scanner]
            for _, row in scan_data.iterrows():
                start_h, start_m = map(int, row['Start Time'].split(':'))
                start_min = (start_h - 8) * 60 + start_m
                
                flag = row['Delay Reason']
                bg = '#e0e0e0'
                for k, v in colors.items():
                    if k in flag: bg = v; break
                
                # Split composite strings to get the primary reason for the label
                primary_flag = flag.split(',')[0] if flag != "None" else "Std"
                label = "Std" if flag == "None" else f"{primary_flag} ({row['Duration']}m)"
                
                tooltip = f"Patient: {row['Patient Number']} | Age: {row['Age']}\nIndication: {row['Clinical Indication']}\nScan: {row['Scan Type']}\nFull Flag: {flag}\nTime: {row['Start Time']}-{row['End Time']}"
                html += f'<div class="slot" style="left:{start_min*1.5}px; width:{row["Duration"]*1.5}px; background-color:{bg};" title="{tooltip}">{label}</div>'
            html += '</div></div>'
        html += '</div>'
    html += "</body></html>"
    with open(filename, "w") as f: f.write(html)
    print(f"HTML Generated: {filename}")

if __name__ == "__main__":
    df = load_and_prep_data()
    if not df.empty:
        schedule = run_scheduler(df)
        df_res = pd.DataFrame(schedule)
        df_res.to_csv(OUTPUT_CSV, index=False)
        generate_html_visualisation(df_res, OUTPUT_HTML)
        
        hoists = df_res[df_res['Delay Reason'].str.contains('Hoist')]
        if not hoists.empty:
            print(f"Total Hoists Booked: {len(hoists)}")
            print(f"Total patients booked: {len(df_res)}")
            print(f"Total patients requested: {len(df)}")
            print(f"Unbooked patients: {len(df) - len(df_res)}")