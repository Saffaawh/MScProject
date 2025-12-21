import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import joblib
import warnings
warnings.filterwarnings("ignore")

REQUESTS_FILE = 'feature_engineered_requests_mri_dataset.csv'
CLASSIFIER_MODEL = 'random_forest_model_pipeline.joblib'
REGRESSOR_MODEL = 'appointment_delay_regression_model.pkl'

START_DATE = date(2025, 1, 2)
END_DATE = date(2025, 3, 31)

MAX_HOISTS_PER_DAY = 2
MAX_CLAUSTRO_SEDATION_PER_DAY = 2

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

def load_and_predict():
    df = pd.read_csv(REQUESTS_FILE)

    clf = joblib.load(CLASSIFIER_MODEL)
    reg = joblib.load(REGRESSOR_MODEL)

    required_cols = [
        'Age', 'Gender', 'Scan Type', 'Mobility Needs', 'Wheelchair User',
        'Hoist Required', 'Poor Mobility', 'Cognitive Conditions',
        'Learning Difficulty', 'Cognitive Impairment', 'Interpreter Required',
        'Sedation needed', 'Contrast', 'Difficult IV Access',
        'Hospital Transport Required'
    ]

    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            df[col] = "No"

    X_raw = df[required_cols]

    df['Predicted_Class'] = clf.predict(X_raw)
    df['Predicted_Overrun'] = reg.predict(X_raw)

    # Compute total duration
    df['Duration'] = df.apply(
        lambda row: scan_types_details.get(row['Scan Type'], 30) + max(0, round(row['Predicted_Overrun'])),
        axis=1
    )

    # Sort by risk + date
    df.sort_values(by=['Predicted_Class', 'Request Date'], ascending=[False, True], inplace=True)

    return df

def is_valid_slot(patient, scanner_name, current_dt, end_dt, daily_stats):

    # Extract flags directly from dataset
    hoist = patient['Hoist Required'] == "Yes"
    sedation = patient['Sedation needed'] == "Yes"
    claustro = False  # You no longer have this flag
    transport = patient['Hospital Transport Required'] == "Yes"

    risk_class = patient['Predicted_Class']
    start_hour = current_dt.hour

    # 1. Class 3 safety rules
    if risk_class == 'Class 3':
        if start_hour >= 17:
            return False

    # 2. Daily quotas
    if hoist and daily_stats['Hoists'] >= MAX_HOISTS_PER_DAY:
        return False

    if sedation and daily_stats['Sedation'] >= MAX_CLAUSTRO_SEDATION_PER_DAY:
        return False

    # 3. Clinical constraints
    if hoist and 12 <= start_hour < 14:
        return False

    if transport:
        if start_hour < 9:
            return False
        if end_dt.hour >= 17 and end_dt.minute > 0:
            return False

    # Scanner-specific constraints
    if sedation and scanner_name == 'MRI Scanner 3':
        return False

    if scanner_name == 'MRI Scanner 2' and hoist:
        return False

    return True
def is_valid_slot(patient, scanner_name, current_dt, end_dt, scanner_stats):
    hoist = patient['Hoist Required'] == "Yes"
    sedation = patient['Sedation needed'] == "Yes"
    transport = patient['Hospital Transport Required'] == "Yes"

    risk_class = patient['Predicted_Class']
    start_hour = current_dt.hour

    # Class 3 cut-off
    if risk_class == 'Class 3' and start_hour >= 17:
        return False

    # Hoists: max 2 per scanner per day
    if hoist and scanner_stats[scanner_name]['Hoists'] >= 2:
        return False

    # Sedation: max 2 in a row on that scanner
    if sedation and scanner_stats[scanner_name]['Sedation_Streak'] >= 2:
        return False

    # Transport window
    if transport:
        if start_hour < 9:
            return False
        if end_dt.hour >= 17 and end_dt.minute > 0:
            return False

    # Scanner-specific rules
    if sedation and scanner_name == 'MRI Scanner 3':
        return False

    if scanner_name == 'MRI Scanner 2' and hoist:
        return False

    return True


def run_scheduler(df):
    schedule = []
    queue = df.to_dict('records')

    current_date = START_DATE

    while queue and current_date <= END_DATE:
        # Per-scanner daily stats
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

            while curr < day_end and queue:
                booked_idx = None

                # 🔁 LOOP THROUGH ALL PATIENTS
                for i, p in enumerate(queue):
                    dur = int(p['Duration'])
                    end = curr + timedelta(minutes=dur)

                    # If it doesn't fit in the day, skip this patient
                    if end > day_end:
                        continue

                    # Check all clinical/operational rules
                    if is_valid_slot(p, scanner_name, curr, end, scanner_stats):
                        booked_idx = i
                        break

                # No patient fits at this time → move time forward and try again
                if booked_idx is None:
                    curr += timedelta(minutes=10)
                    continue

                # We found a patient that fits
                p = queue.pop(booked_idx)

                # Build flags string for output
                flags = []
                if p['Hoist Required'] == "Yes":
                    flags.append("Hoist")
                if p['Sedation needed'] == "Yes":
                    flags.append("Sedation")
                if p['Hospital Transport Required'] == "Yes":
                    flags.append("Transport")
                if p['Wheelchair User'] == "Yes":
                    flags.append("Wheelchair")

                schedule.append({
                    'Scanner': scanner_name,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Start Time': curr.strftime('%H:%M'),
                    'End Time': end.strftime('%H:%M'),
                    'Patient Number': p['Patient Number'],
                    'Scan Type': p['Scan Type'],
                    'Flags': ", ".join(flags) if flags else "None",
                    'Predicted_Overrun': round(p['Predicted_Overrun'], 1),
                    'Risk_Class': p['Predicted_Class'],
                    'Duration': dur
                })

                # Update per-scanner stats
                if p['Hoist Required'] == "Yes":
                    scanner_stats[scanner_name]['Hoists'] += 1

                if p['Sedation needed'] == "Yes":
                    scanner_stats[scanner_name]['Sedation_Streak'] += 1
                else:
                    scanner_stats[scanner_name]['Sedation_Streak'] = 0

                # Move scanner clock forward by duration + 5-min turnaround
                curr = end + timedelta(minutes=5)

            # Save final availability (not strictly needed but tidy)
            scanner_avail[scanner_name] = curr

        current_date += timedelta(days=1)

    return schedule


def generate_html_visualisation(df_sched, filename):
    colors = {
        'Hoist': '#ff4d4d',
        'Transport': '#ff9933',
        'Wheelchair': '#ffff66',
        'Sedation': '#cc99ff',
        'Interpreter': '#66b3ff',
        'None': '#99ff99'
    }

    # Create Day column from Date
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
        </style></head><body><h1>MRI Schedule Visualisation (ML)</h1><div class="legend">"""

    # Legend
    for label, color in colors.items():
        html += f'<span style="background-color: {color}">{label}</span>'
    html += '</div>'

    # Show first 7 days
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

                # Determine colour
                first_flag = row['Flags'].split(",")[0] if row['Flags'] != "None" else "None"
                bg = colors.get(first_flag, "#e0e0e0")

                label = first_flag if first_flag != "None" else "Std"

                tooltip = (
                    f"Patient: {row['Patient Number']}\n"
                    f"Scan: {row['Scan Type']}\n"
                    f"Flags: {row['Flags']}\n"
                    f"Risk Class: {row['Risk_Class']}\n"
                    f"Predicted Overrun: +{row['Predicted_Overrun']}m\n"
                    f"Duration: {row['Duration']}m\n"
                    f"Time: {row['Start Time']} - {row['End Time']}"
                )

                html += (
                    f'<div class="slot" '
                    f'style="left:{start_min*1.5}px; width:{row["Duration"]*1.5}px; background-color:{bg};" '
                    f'title="{tooltip}">{label}</div>'
                )

            html += '</div></div>'  # close timeline + scanner-row

        html += '</div>'  # close day-container

    html += "</body></html>"

    with open(filename, "w") as f:
        f.write(html)

    print(f"HTML Generated: {filename}")

if __name__ == "__main__":
    df_new = load_and_predict()

    print("Total patients:", len(df_new))
    print("Duration summary:\n", df_new['Duration'].describe())
    print("Predicted classes:\n", df_new['Predicted_Class'].value_counts())
    print("Flags counts:",
          "Hoist", (df_new['Hoist Required'] == "Yes").sum(),
          "Sed", (df_new['Sedation needed'] == "Yes").sum(),
          "Transport", (df_new['Hospital Transport Required'] == "Yes").sum())
    
    schedule = run_scheduler(df_new)
    print("Booked appointments:", len(schedule))
    
    df_sched = pd.DataFrame(schedule)
    df_sched.to_csv("final_ml_schedule.csv", index=False)

    generate_html_visualisation(df_sched, "mri_schedule_visualisation_ml.html")

    print("Scheduling complete.")
    print("Appointments booked:", len(df_sched))