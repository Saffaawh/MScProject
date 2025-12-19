#This code will schedule the new MRI appointments into available slots based on working hours, weekends, and holidays.
#It ensures no appointments are scheduled on Christmas Day and handles different working hours for weekdays, weekends, and bank holidays.
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, date
from workalendar.europe import UnitedKingdom
import os
import sys
# Logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Time Definitions ---
UK_CALENDAR = UnitedKingdom()
START_DATE = datetime(2025, 1, 1) 
END_DATE = datetime(2025, 6, 30) # Extended to 6 months to fix capacity
TIME_SLOT_INCREMENT_MINUTES = 15 

# --- Scanner & Scan Definitions ---
scan_types_details = {
    'Brain': {'ScheduledTime': 30, 'Contrast' : False},
    'Brain with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Cervical Spine': {'ScheduledTime': 30, 'Contrast' : False},
    'Cervical Spine with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Thoracic Spine': {'ScheduledTime': 30, 'Contrast' : False},
    'Thoracic Spine with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Lumbar Spine': {'ScheduledTime': 30, 'Contrast' : False}, 
    'Lumbar Spine with Contrast': {'ScheduledTime': 45, 'Contrast' : True},
    'Whole Spine': {'ScheduledTime': 45, 'Contrast' : False},
    'Whole Spine with Contrast': {'ScheduledTime': 60, 'Contrast' : True}}

scanner_details = {
    'MRI Scanner 1': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': ['Wide Bore']},
    'MRI Scanner 2': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': []},
    'MRI Scanner 3': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': ['Small Bore']},
    'MRI Scanner 4': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': ['Research Only', 'Closed Wednesdays']}
}   

# --- Constraint Definitions ---
EXCLUSION_MAPPING = {
    'Mobility Needs': ['MRI Scanner 2'],
    'Wheelchair User': ['MRI Scanner 2'], 
    'Claustrophobic': ['MRI Scanner 3'], 
    'Hospital Transport Required': ['MRI Scanner 4']
}
PREFERENCE_MAPPING = {
    'Claustrophobic': ['MRI Scanner 1']
}

FLAG_COLUMNS_SOURCE = [
    'Mobility Needs', 'Wheelchair User', 'Hoist Required',
    'Cognitive Conditions', 'Learning Difficulty', 'Cognitive Impairment',
    'Interpreter Required', 'Sedation needed', 
    'Difficult IV Access', 'Hospital Transport Required'
]


# --- 1. Load Data ---

try:
    df_requests = pd.read_csv('final_balanced_100_requests.csv')
    print(f"Loaded {len(df_requests)} balanced MRI requests.")
except FileNotFoundError:
    print("ERROR: final_balanced_100_requests.csv not found. Please run Script 2 first.")
    sys.exit(1)

try:
    df_overrun = pd.read_csv('Average_predicted_Overruns_by_flag.csv')
    overrun_dict = df_overrun.set_index('Flag')['Average Predicted Overrun Minutes'].to_dict()
    print("Loaded average predicted overruns by flag.")
except FileNotFoundError:
    print("ERROR: Average_predicted_Overruns_by_flag.csv not found. Please run Regression Model code first.")
    sys.exit(1)

# Load the original raw dataset to pull the 'Delay Reason' column 
try:
    df_raw = pd.read_csv('historical_mri_appointment_dataset_raw.csv') 
    df_delay_reason = df_raw[['PatientNumber', 'Delay Reason']].drop_duplicates()
    df_requests = pd.merge(df_requests, df_delay_reason, on='PatientNumber', how='left')
    print("Successfully merged raw 'Delay Reason' column.")
except (FileNotFoundError, KeyError):
    print("WARNING: Could not merge original 'Delay Reason' column. 'Original Delay Reason' will be 'N/A'.")
    df_requests['Delay Reason'] = 'N/A'


# --- 2. Calculate Final Appointment Duration for each Request ---

FLAG_COLUMNS = [col for col in FLAG_COLUMNS_SOURCE if col in df_requests.columns]

def calculate_predicted_duration(row, overrun_dict, scan_types_details):
    scan_type = row['Scan Type']
    base_time = scan_types_details.get(scan_type, {}).get('ScheduledTime', 30) 
    max_overrun = 0
    active_flags = []
    
    # Check individual patient flags (Robust Check for String/Float/'Yes'/'No')
    for flag in FLAG_COLUMNS:
        flag_value = row.get(flag)
        # Check for various forms of "True" (1, 1.0, 'Yes', 'True')
        flag_str = str(flag_value).strip().lower()
        if flag_str in ['1', '1.0', 'yes', 'true']:
            active_flags.append(flag)
            
    # Check for Contrast flag
    requires_contrast_value = row.get('Requires Contrast', 0)
    try:
        if pd.notna(requires_contrast_value) and np.isclose(float(requires_contrast_value), 1.0):
            active_flags.append('Contrast')
        elif pd.notna(requires_contrast_value) and isinstance(requires_contrast_value, bool) and requires_contrast_value:
             active_flags.append('Contrast')
    except (ValueError, TypeError):
        pass

    # Determine the overrun based on the flag with the highest average delay
    for flag in active_flags:
        overrun = overrun_dict.get(flag, 0)
        max_overrun = max(max_overrun, overrun)
        
    # Calculate the raw total duration
    raw_total_duration = base_time + max_overrun
    
    # FIX 2: Round the total duration UP to the nearest multiple of 5
    M = 5 
    predicted_total_duration = M * np.ceil(raw_total_duration / M)
        
    return int(predicted_total_duration)

# Apply the duration calculation (Creates the 'Predicted Total Duration' column)
df_requests['Predicted Total Duration'] = df_requests.apply(
    lambda row: calculate_predicted_duration(row, overrun_dict, scan_types_details), axis=1
)

df_requests = df_requests.sort_values(by='RequestID').reset_index(drop=True)
print(f"Calculated predicted durations. Max duration: {df_requests['Predicted Total Duration'].max():.0f} min.")


# --- 3. Generate All Available Time Slots ---

def generate_slots(start_date, end_date, slot_increment_minutes, scanner_details):
    """Generates all available slots for all scanners between two dates."""
    all_slots = {scanner: [] for scanner in scanner_details.keys()}
    current_date = start_date
    uk_holidays = UK_CALENDAR.holidays(start_date.year) + UK_CALENDAR.holidays(end_date.year)
    
    while current_date <= end_date:
        current_date_date = current_date.date()
        if current_date_date == date(current_date.year, 12, 25):
            current_date += timedelta(days=1)
            continue
            
        is_weekend = current_date.weekday() >= 5
        is_bank_holiday = current_date_date in uk_holidays

        for scanner_name, details in scanner_details.items():
            if is_weekend or is_bank_holiday:
                start_hour, end_hour = details['WeekendHours']
            else: 
                start_hour, end_hour = details['WeekdayHours']
            
            if scanner_name == 'MRI Scanner 4' and current_date.weekday() == 2:
                 continue

            current_time = datetime(current_date.year, current_date.month, current_date.day, start_hour, 0)
            end_of_day = datetime(current_date.year, current_date.month, current_date.day, end_hour, 0)
            
            while current_time < end_of_day:
                slot_end = current_time + timedelta(minutes=slot_increment_minutes)
                is_lunch_hour = (current_time.hour >= 12 and current_time.hour < 14)
                
                is_hoist_slot = not is_weekend and not is_bank_holiday and \
                                not is_lunch_hour and \
                                current_time.hour >= 9 and current_time.hour < 17

                is_wheelchair_slot = not is_weekend and not is_bank_holiday

                all_slots[scanner_name].append({
                    'Start': current_time,
                    'End': slot_end, 
                    'Available': True,
                    'Duration': slot_increment_minutes,
                    'IsWeekend': is_weekend,
                    'IsBankHoliday': is_bank_holiday,
                    'IsHoistSuitable': is_hoist_slot,
                    'IsWheelchairSuitable': is_wheelchair_slot
                })
                current_time = slot_end
        
        current_date += timedelta(days=1)
        
    return all_slots

appointment_slots = generate_slots(START_DATE, END_DATE, TIME_SLOT_INCREMENT_MINUTES, scanner_details)


# --- 4. Scheduling Algorithm (FCFS with Constraints & Priority) ---

scheduled_appointments = []
unassigned_requests = []

# Simple FCFS iteration
for req_index, request in df_requests.iterrows():
    duration = request['Predicted Total Duration']
    
    # 1. INITIALIZE VARIABLES (Reset for each request)
    best_slot = None 
    best_scanner = None
    best_start_time = datetime(2100, 1, 1)

    historical_class = request.get('Historical Class') 
    requires_hoist = request.get('Hoist Required', 0) == 1
    requires_wheelchair = request.get('Wheelchair User', 0) == 1
    
    # Check if the patient has ANY flag for general categorization
    has_any_flag = (requires_hoist or requires_wheelchair or 
                    request.get('Mobility Needs', 0) == 1 or 
                    request.get('Cognitive Conditions', 0) == 1 or 
                    request.get('Hospital Transport Required', 0) == 1)


    # 4a & 4b. Identify Suitable Scanners & Preference
    suitable_scanners = list(scanner_details.keys())
    
    for flag, excluded_scanners in EXCLUSION_MAPPING.items():
        if flag in request and request[flag] == 1:
            suitable_scanners = [s for s in suitable_scanners if s not in excluded_scanners]

    preferred_scanner = None
    if request.get('Claustrophobic', 0) == 1 and 'MRI Scanner 1' in suitable_scanners:
         preferred_scanner = 'MRI Scanner 1'
    
    # 4c. Find the Earliest Available Slot (Priority Logic)
    
    # Set priority: Complex patients need Weekday. Simple patients go to Weekend first.
    if requires_hoist or requires_wheelchair:
        slots_to_prioritize = ['Weekday']
    elif not has_any_flag:
        slots_to_prioritize = ['Weekend', 'Weekday']
    else: # Other complex flags 
        slots_to_prioritize = ['Weekday', 'Weekend']


    for slot_type in slots_to_prioritize:
        
        # If we already found a slot in the previous priority group, stop looking.
        if best_slot:
            break

        scanners_to_check = []
        if preferred_scanner:
            scanners_to_check.append(preferred_scanner)
        scanners_to_check.extend([s for s in suitable_scanners if s != preferred_scanner])

        # We iterate through ALL suitable scanners for this priority (e.g. Weekday)
        # We track the absolute best start time found across all of them.
        for scanner_name in scanners_to_check:
            if scanner_name not in appointment_slots: continue
                
            scanner_slots = appointment_slots[scanner_name]
            
            # Find the earliest slot on THIS scanner
            for i in range(len(scanner_slots)):
                current_slot = scanner_slots[i]
                start_time = current_slot['Start']
                
                # Filter by Slot Type (Weekday/Weekend Priority)
                if slot_type == 'Weekend' and not current_slot['IsWeekend']: continue
                if slot_type == 'Weekday' and current_slot['IsWeekend']: continue
                
                # Optimization: If this slot starts later than a slot we ALREADY found on another scanner, skip it.
                if start_time >= best_start_time:
                    break 

                # Check required duration
                num_slots_needed = int(np.ceil(duration / TIME_SLOT_INCREMENT_MINUTES))
                end_index = i + num_slots_needed
                
                if end_index > len(scanner_slots): break
                    
                required_slots = scanner_slots[i:end_index]
                
                # --- Constraint Checks ---
                if not all(slot['Available'] for slot in required_slots): continue
                if requires_hoist and not all(slot['IsHoistSuitable'] for slot in required_slots): continue
                if requires_wheelchair and not all(slot['IsWheelchairSuitable'] for slot in required_slots): continue

                # --- Decision: Compare against Best Found Time ---
                # Since we checked "start_time >= best_start_time" above, if we are here, this is strictly better.
                if start_time < best_start_time:
                    best_start_time = start_time
                    best_scanner = scanner_name
                    best_slot = (i, end_index)
                
                # Found the earliest valid slot on THIS scanner, move to the next scanner to see if it has an even earlier one
                break 


    # 4d. Schedule the Appointment 
    if best_slot:
        start_index, end_index = best_slot
        
        # Mark slots as unavailable
        for i in range(start_index, end_index):
            appointment_slots[best_scanner][i]['Available'] = False
            
        appointment_end_time = best_start_time + timedelta(minutes=duration)
        
        # Capture active flags (Robust check)
        active_flags_list = []
        for flag in FLAG_COLUMNS:
            flag_value = request.get(flag)
            # Check for various forms of "True" (1, 1.0, 'Yes', 'True')
            flag_str = str(flag_value).strip().lower()
            if flag_str in ['1', '1.0', 'yes', 'true']:
                active_flags_list.append(flag)
        
        scheduled_appointments.append({
            'RequestID': request['RequestID'],
            'PatientNumber': request['PatientNumber'],
            'Scheduled Scanner': best_scanner,
            'Scheduled Start Time': best_start_time,
            'Scheduled End Time': appointment_end_time,
            'Predicted Duration (min)': duration,
            'Scan Type': request['Scan Type'],
            
            'Patient Flags Present': ', '.join(active_flags_list) if active_flags_list else 'None',
            'Historical Class': historical_class,
            'Original Delay Reason': request.get('Delay Reason', 'N/A'),
            'Clinical Indication': request.get('Clinical Indication')
        })
    else:
        unassigned_requests.append(request['RequestID'])

## --- 5. Final Output ---

df_schedule = pd.DataFrame(scheduled_appointments)

# 1. Ensure the source columns are datetime objects
df_schedule['Scheduled Start Time'] = pd.to_datetime(df_schedule['Scheduled Start Time'])
df_schedule['Scheduled End Time'] = pd.to_datetime(df_schedule['Scheduled End Time'])

# 2. Create NEW separate Date and Time columns
# We use strftime('%H:%M:%S') for time to ensure Power BI recognizes it strictly as Time.

# Start Split
df_schedule['Scheduled Date'] = df_schedule['Scheduled Start Time'].dt.date
df_schedule['Scheduled Time'] = df_schedule['Scheduled Start Time'].dt.strftime('%H:%M:%S')

# End Split
df_schedule['Scheduled End Date'] = df_schedule['Scheduled End Time'].dt.date
df_schedule['Scheduled End Time Only'] = df_schedule['Scheduled End Time'].dt.strftime('%H:%M:%S')

# 3. Sort and Save
df_schedule = df_schedule.sort_values(by='Scheduled Start Time').reset_index(drop=True)

print("\n=======================================================")
print("✅ Scheduling Complete")
print("=======================================================")
print(f"Total Requests Processed: {len(df_requests)}")
print(f"Total Appointments Scheduled: {len(df_schedule)}")
print(f"Total Unassigned Requests: {len(unassigned_requests)}")

if not df_schedule.empty:
    print("\nFirst 5 Scheduled Appointments:")
    print(df_schedule[['Scheduled Date', 'Scheduled Time', 'Scheduled End Time Only']].head())

    df_schedule.to_csv('mri_appointment_schedule.csv', index=False)
    print("\nSchedule saved to 'mri_appointment_schedule.csv'")

# --- 6. Visualization (Timeline with Class Colors & Date Dropdown) ---

import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# 1. Prepare Data
df_viz = df_schedule.sort_values(by=['Scheduled Scanner', 'Scheduled Start Time']).copy()

# Force 'Historical Class' to be a string so the Legend shows distinct colors (0, 1, 2, 3)
df_viz['Historical Class'] = df_viz['Historical Class'].astype(str)

# Create a helper column for the date string (used for the dropdown logic)
df_viz['Date_Str'] = df_viz['Scheduled Start Time'].dt.strftime('%Y-%m-%d')

# 2. Create the Base Timeline Chart
fig = px.timeline(
    df_viz, 
    x_start="Scheduled Start Time", 
    x_end="Scheduled End Time", 
    y="Scheduled Scanner",
    color="Historical Class",  # <--- Logic 1: Color by Class
    hover_data={
        "PatientNumber": True,
        "Scan Type": True,
        "Predicted Duration (min)": True,
        "Patient Flags Present": True,
        "Original Delay Reason": True,
        "Historical Class": False,
        "Scheduled Scanner": False,
        "Date_Str": False
    },
    title="MRI Appointment Schedule (Filter by Day)",
    labels={"Historical Class": "Patient Class"}, 
    template="plotly_white"
)

# 3. Create the Date Filter Dropdown Logic
# Get all unique dates from the schedule
unique_dates = df_viz['Date_Str'].unique()
unique_dates.sort()

# Create a list of buttons for the dropdown
buttons = []

# Button 1: "Show All" (Resets the view)
buttons.append(dict(
    method="relayout",
    label="Show All",
    args=[{"xaxis.range": [None, None]}]
))

# Create a button for every single date in the schedule
for date_str in unique_dates:
    # Define the start and end of that specific day for the zoom
    # We set the range from 07:00 to 21:00 to cover the shifts comfortably
    start_range = f"{date_str} 07:00"
    end_range = f"{date_str} 21:00"
    
    buttons.append(dict(
        method="relayout",
        label=date_str,
        args=[{"xaxis.range": [start_range, end_range]}]
    ))

# 4. Update Layout with the Dropdown and Formatting
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.15, # Position of the dropdown
            xanchor="left",
            y=1.15,
            yanchor="top"
        )
    ],
    xaxis_title="Time of Day",
    yaxis_title="MRI Scanner",
    showlegend=True,
    height=600,
    xaxis=dict(
        tickformat="%H:%M",
        tickmode="linear",
        dtick=3600000.0 * 1, # Show a tick every 1 hour
    )
)

# Add borders to blocks for clarity
fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

# 5. Save
fig.write_html("MRI_Schedule_Dashboard_Final.html")
print("Dashboard saved as 'MRI_Schedule_Dashboard_Final.html'")