import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---

REQUESTS_FILE = 'sampled_mri_requests_dataset.csv'
OUTPUT_CSV = 'final_baseline_schedule.csv'
OUTPUT_HTML = 'mri_schedule_visualisation_baseline.html'

START_DATE = date(2025, 1, 2)
END_DATE = date(2025, 3, 31)

# Standard "Rule Book" Times (The Baseline)
STANDARD_SCAN_TIMES = {
    'Brain': 30, 'Brain with Contrast': 45,
    'Cervical Spine': 30, 'Cervical Spine with Contrast': 45,
    'Thoracic Spine': 30, 'Thoracic Spine with Contrast': 45,
    'Lumbar Spine': 30, 'Lumbar Spine with Contrast': 45,
    'Whole Spine': 45, 'Whole Spine with Contrast': 60
}

# Scanner Opening Hours
scanners = {
    'MRI Scanner 1': {'Start': 8, 'End': 20},
    'MRI Scanner 2': {'Start': 8, 'End': 20},
    'MRI Scanner 3': {'Start': 8, 'End': 20},
    'MRI Scanner 4': {'Start': 8, 'End': 20}
}

# Columns to check for flagging (just for color in visualization)
FLAG_COLUMNS = [
    'Hoist Required', 'Sedation needed', 'Hospital Transport Required', 
    'Wheelchair User', 'Poor Mobility', 'Cognitive Conditions', 
    'Learning Difficulty', 'Cognitive Impairment', 'Interpreter Required', 
    'Difficult IV Access'
]

# --- 2. DATA PREPARATION ---

def load_and_prep_data():
    if not os.path.exists(REQUESTS_FILE):
        logger.error(f"File not found: {REQUESTS_FILE}")
        return pd.DataFrame()

    df_patients = pd.read_csv(REQUESTS_FILE)

    if 'PatientNumber' in df_patients.columns:
        df_patients.rename(columns={'PatientNumber': 'Patient Number'}, inplace=True)

    # Construct 'Delay Reason' string for the Visualization (so you can see the risks)
    def construct_reason(row):
        reasons = []
        for col in FLAG_COLUMNS:
            if col in row and row[col] == "Yes":
                reasons.append(col)
        return ", ".join(reasons) if reasons else "None"

    df_patients['Delay Reason'] = df_patients.apply(construct_reason, axis=1)

    # --- BASELINE LOGIC: ASSIGN STANDARD DURATION ONLY ---
    # No buffers, no adjustments. Just the "Menu Price".
    df_patients['Scheduled_Duration'] = df_patients['Scan Type'].map(STANDARD_SCAN_TIMES).fillna(30).astype(int)
    
    # Sort by request date (Standard First-Come-First-Served)
    df_patients.sort_values(by=['Request Date'], inplace=True)
    
    return df_patients

# --- 3. CONSTRAINT CHECKS ---

def apply_scanner_closures(scanner_name, current_dt):
    """
    Enforces scanner closures.
    Rule: Scanner 4 closed Wed 13:00-17:00 (Research block).
    """
    if scanner_name == 'MRI Scanner 4' and current_dt.weekday() == 2: # Wednesday
        if 13 <= current_dt.hour < 17:
            # Jump to 17:00
            return current_dt.replace(hour=17, minute=0, second=0, microsecond=0)
    return current_dt

# --- 4. SCHEDULER ENGINE (BASELINE) ---

def run_baseline_scheduler(df_patients):
    schedule_log = []
    queue = df_patients.to_dict('records')

    current_date = START_DATE

    print(f"Scheduling {len(queue)} patients (Baseline Rules)...")

    while queue and current_date <= END_DATE:
        
        # Reset scanner availability for the new day
        scanner_avail = {
            s: datetime.combine(current_date, time(8, 0))
            for s in scanners
        }

        day_end = datetime.combine(current_date, time(20, 0))

        # Iterate through scanners
        for scanner_name in scanners:
            curr = scanner_avail[scanner_name]
            curr = apply_scanner_closures(scanner_name, curr)

            while curr < day_end and queue:
                # Re-check closure inside the loop (in case we crossed into a closure block)
                curr = apply_scanner_closures(scanner_name, curr)
                
                if curr >= day_end:
                    break

                # Pop the next patient
                p = queue[0] # Look at next patient
                dur = int(p['Scheduled_Duration'])
                end = curr + timedelta(minutes=dur)

                # Check 1: Does it fit before closing time?
                if end > day_end:
                    # Scanner full for the day, break to next scanner
                    break
                
                # Check 2: Does it fit before a closure block (e.g. Wed 1pm)?
                # If we are Scanner 4 on Wednesday, and 'end' is inside the 1pm-5pm block
                if scanner_name == 'MRI Scanner 4' and current_date.weekday() == 2:
                     closure_start = datetime.combine(current_date, time(13,0))
                     if curr < closure_start and end > closure_start:
                         # Fits physically, but hits the wall. Move 'curr' to 17:00
                         curr = datetime.combine(current_date, time(17,0))
                         continue # Try again at 17:00

                # --- BOOKING CONFIRMED ---
                # Remove from queue
                p = queue.pop(0)

                schedule_log.append({
                    'Scanner': scanner_name,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Day': current_date.strftime('%A'),
                    'Start Time': curr.strftime('%H:%M'),
                    'End Time': end.strftime('%H:%M'),
                    'Patient Number': p['Patient Number'],
                    'Age': p['Age'],
                    'Clinical Indication': p.get('Clinical Indication', 'Unknown'),
                    'Delay Reason': p['Delay Reason'],
                    'Scan Type': p['Scan Type'],
                    'Duration': dur
                })

                # Move time forward + small changeover gap (e.g. 5 mins standard efficiency)
                curr = end + timedelta(minutes=5)
            
            scanner_avail[scanner_name] = curr

        current_date += timedelta(days=1)

    return schedule_log

# --- 5. VISUALISATION ---

def generate_html_visualisation(df_sched, filename):
    # Color map for flags (Visual aid only - does not affect scheduling logic)
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
        </style></head><body><h1>Baseline MRI Schedule (Standard Rules)</h1><div class="legend">"""
    
    seen = []
    for r, c in colors.items():
        if c not in seen:
            lbl = "Sedation / Claustrophobia" if "Sedation" in r else r
            html += f'<span style="background-color: {c}">{lbl}</span>'
            seen.append(c)
    html += '<span style="background-color: #e0e0e0">Other</span></div>'
    
    if df_sched.empty:
        print("Warning: Schedule is empty.")
        return

    dates = sorted(df_sched['Date'].unique())[:7] # Show first week
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
                
                primary_flag = flag.split(',')[0] if flag != "None" else "Std"
                label = "Std" if flag == "None" else f"{primary_flag} ({row['Duration']}m)"
                
                tooltip = f"Patient: {row['Patient Number']} | Scan: {row['Scan Type']}\nFlag: {flag}\nBooked: {row['Start Time']}-{row['End Time']}"
                html += f'<div class="slot" style="left:{start_min*1.5}px; width:{row["Duration"]*1.5}px; background-color:{bg};" title="{tooltip}">{label}</div>'
            html += '</div></div>'
        html += '</div>'
    html += "</body></html>"
    with open(filename, "w") as f: f.write(html)
    print(f"HTML Generated: {filename}")

if __name__ == "__main__":
    df = load_and_prep_data()
    if not df.empty:
        schedule = run_baseline_scheduler(df)
        df_res = pd.DataFrame(schedule)
        
        # Save results
        df_res.to_csv(OUTPUT_CSV, index=False)
        print(f"Schedule CSV saved to: {OUTPUT_CSV}")
        
        generate_html_visualisation(df_res, OUTPUT_HTML)
        
        print(f"Total patients booked: {len(df_res)}")
        print(f"Total patients requested: {len(df)}")