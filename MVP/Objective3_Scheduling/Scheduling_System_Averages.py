import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 1. RESTORE MISSING AVERAGES FILE ---
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

# --- NEW: SAFETY LIMITS ---
MAX_HOISTS_PER_DAY = 2             # Manual Handling Safety Limit
MAX_CLAUSTRO_SEDATION_PER_DAY = 2  # Resource/Staff Stress Limit

# --- NEW: END OF DAY BUFFER ---
# No complex patient (Class 2 or 3) can start after this time.
# Scanners close at 20:00. 17:30 gives a 2.5hr safety buffer.
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
    'MRI Scanner 1': {'Type': 'Wide Bore', 'Start': 8, 'End': 20, 'WeekendStart': 9, 'WeekendEnd': 17},
    'MRI Scanner 2': {'Type': 'Standard',  'Start': 8, 'End': 20, 'WeekendStart': 9, 'WeekendEnd': 17},
    'MRI Scanner 3': {'Type': 'Small Bore', 'Start': 8, 'End': 20, 'WeekendStart': 9, 'WeekendEnd': 17},
    'MRI Scanner 4': {'Type': 'Research',   'Start': 8, 'End': 20, 'WeekendStart': 9, 'WeekendEnd': 17}
}

# --- 3. DATA PREPARATION ---

def load_and_prep_data():
    if not os.path.exists(REQUESTS_FILE):
        logger.error(f"File not found: {REQUESTS_FILE}")
        return pd.DataFrame()

    df_patients = pd.read_csv(REQUESTS_FILE)
    df_patients['Delay Reason'] = df_patients['Delay Reason'].fillna('None')
    
    df_avg = pd.read_csv(AVERAGES_FILE)
    avg_map = dict(zip(df_avg['Flag'].str.lower(), df_avg['Average Predicted Overrun Minutes']))

    def get_extra_time(flag_str, scan_type):
        flag_str = str(flag_str).lower()
        extra = 0
        if 'hoist' in flag_str: extra = max(extra, avg_map.get('hoist required', 38))
        if 'transport' in flag_str: extra = max(extra, avg_map.get('hospital transport required', 38))
        if 'wheelchair' in flag_str: extra = max(extra, avg_map.get('wheelchair user', 22))
        if 'mobility' in flag_str: extra = max(extra, avg_map.get('poor mobility', 20))
        if 'sedation' in flag_str or 'claustro' in flag_str: 
            extra = max(extra, avg_map.get('sedation needed', 34))
        if 'interpreter' in flag_str: extra = max(extra, avg_map.get('interpreter required', 20))
        if 'cognitive' in flag_str: extra = max(extra, avg_map.get('cognitive impairment', 18))
        if 'learning' in flag_str: extra = max(extra, avg_map.get('learning difficulty', 19))
        if 'diva' in flag_str or 'iv' in flag_str: extra = max(extra, avg_map.get('difficult iv access', 16))
        if 'contrast' in scan_type.lower():
             extra += avg_map.get('contrast', 8)
        return round(extra)

    predicted_durations = []
    for _, row in df_patients.iterrows():
        base = scan_types_details.get(row['Scan Type'], 30)
        overrun = get_extra_time(row['Delay Reason'], row['Scan Type'])
        predicted_durations.append(base + overrun)
        
    df_patients['Predicted_Duration'] = predicted_durations
    
    # Priority Sorting
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
    
    # Identify Complex Patients (Any flagged patient essentially)
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
        if current_dt.weekday() == 2 and start_hour >= 12: return False
            
    return True

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

        # End of day
        day_end = datetime.combine(current_date, time(20, 0))

        # Loop through scanners
        for scanner_name in scanners:

            curr = scanner_avail[scanner_name]

            while curr < day_end and queue:

                booked_idx = None

                # 🔁 LOOP THROUGH ALL PATIENTS
                for i, p in enumerate(queue):

                    flag = str(p['Delay Reason'])
                    dur = int(p['Predicted_Duration'])
                    end = curr + timedelta(minutes=dur)

                    # If it doesn't fit in the day, skip
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

                    # If we reach here → patient fits
                    booked_idx = i
                    break

                # If no patient fits → move time forward
                if booked_idx is None:
                    curr += timedelta(minutes=10)
                    continue

                # Book the patient
                p = queue.pop(booked_idx)

                schedule_log.append({
                    'Scanner': scanner_name,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Day': current_date.strftime('%A'),
                    'Start Time': curr.strftime('%H:%M'),
                    'End Time': end.strftime('%H:%M'),
                    'Patient Number': p['Patient Number'],
                    'Age': p['Age'],
                    'Clinical Indication': p['Type of Issue'],
                    'Delay Reason': flag,
                    'Scan Type': p['Scan Type'],
                    'Duration': dur
                })

                # Update scanner stats
                if 'Hoist' in flag:
                    scanner_stats[scanner_name]['Hoists'] += 1

                if 'Sedation' in flag or 'Claustro' in flag:
                    scanner_stats[scanner_name]['Sedation_Streak'] += 1
                else:
                    scanner_stats[scanner_name]['Sedation_Streak'] = 0

                # Move scanner forward
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
                
                label = "Std" if flag == "None" else f"{flag.split(' ')[0]} ({row['Duration']}m)"
                tooltip = f"Patient: {row['Patient Number']} | Age: {int(row['Age'])}\nIndication: {row['Clinical Indication']}\nScan: {row['Scan Type']}\nFull Flag: {flag}\nTime: {row['Start Time']}-{row['End Time']}"
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
            print("Check Date Distribution (Top 5 Days):")
            print(hoists['Date'].value_counts().head())
            print(f"Total patients booked: {len(df_res)}")
print(f"Total patients requested: {len(df)}")
print(f"Unbooked patients: {len(df) - len(df_res)}")