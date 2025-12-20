# Objective 1: Data Creation
import pandas as pd
import numpy as np
from faker import Faker 
import random
from datetime import datetime, timedelta
 
fake = Faker()
fake.seed_instance(42)

fake = Faker()

num_records = 5000

genders = ['Male','Female', 'Other']
ages = np.random.randint(18, 90, size=num_records)

type_of_problem_distribution = {
    'None': 0.68, # 64% of patients have no issues
    'Wheelchair user':0.13, #81
    'Hoist required':0.01, #82
    'DIVA': 0.01,  #83
    'Cognitive impairment':0.03, #86
    'Claustrophobic': 0.015, #87.5
    'Learning difficulty': 0.01, #88.5
    'Interpreter required': 0.05,#93.5
    'Poor mobility': 0.015,#95
    'Hospital transport': 0.05  #100
}


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

scan_type_names = list(scan_types_details.keys())

from datetime import datetime, timedelta, date
from workalendar.europe import UnitedKingdom
cal = UnitedKingdom()
uk_holidays = cal.holidays(2023) + cal.holidays(2024)

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
appointment = []
current_date = start_date 
while current_date <= end_date:
    if current_date == date(2023, 12, 25) or current_date == date(2024, 12, 25):
        current_date +- timedelta(days=1)
        continue
    is_bank_holiday = current_date in uk_holidays
    is_weekday = current_date.weekday() <5 and current_date not in uk_holidays
    if is_weekday:
        sart_hour , end_hour = 8 , 20 
    elif is_bank_holiday: 
        start_hour, end_hour = 9, 17
    else: #weekend
        start_hour, end_hour = 9, 17

    for hour in range(start_hour, end_hour):
        for minute in [0, 15, 30, 45]:
          appointment.append(datetime(current_date.year , current_date.month, current_date.day, hour, minute))

    current_date += timedelta(days=1)

if len(appointment) < num_records:
    raise ValueError("Not enough appointment slots to schedule all records.")

selected_appointment = random.sample(appointment, num_records)
selected_appointment.sort() 

data = []
for i in range(num_records):
    sceduled_datetime = selected_appointment[i]
    patient_number = str(i + 1).zfill(4)
    age = np.random.choice(ages)
    gender = np.random.choice(genders)

    type_of_issue = np.random.choice(
        list(type_of_problem_distribution.keys()),
        p=list(type_of_problem_distribution.values())
    )

    scan_name = np.random.choice(scan_type_names)
    scheduled_scan_time = scan_types_details[scan_name]['ScheduledTime']
    arrival_time_minutes = random.choice([-20, -15, -10, -5]) #this dataset does not include delays due to patient arriving late 

    if type_of_issue == 'None':
        overrun_minutes = 0 
    else: 
        if type_of_issue in ['Hospital transport', 'Hoist required', 'Claustrophobic']: #these are the delays that usually are over 20 minutes 
            delay_category = np.random.choice(['minor_delay', 'major_delay'], p=[0.1, 0.9])
        else:
            delay_category = np.random.choice(['minor_delay', 'major_delay'], p=[0.7, 0.3])
        if delay_category == 'minor_delay':
            overrun_minutes = random.randint(5, 20)
        else:
            overrun_minutes = random.randint(21, 60)
        
    actual_scan_duration = scheduled_scan_time + overrun_minutes
    actual_datetime = sceduled_datetime + timedelta(minutes=actual_scan_duration)

    data.append([ 
        patient_number,
        age,
        gender, 
        scan_name,
        scheduled_scan_time, 
        arrival_time_minutes,
        actual_scan_duration,
        overrun_minutes,
        sceduled_datetime,
        type_of_issue,
        actual_datetime
    ])

    columns = [
        'PatientNumber',
        'Age',
        'Gender',
        'Scan Type',
        'Scheduled Scan Time',
        'Arrival Time' , 
        'Actual Scan Duration',
        'Overrun Minutes',
        'Scheduled Date and Time',
        'Delay Reason', 
        'Actual End Time'
    ]
historical_df = pd.DataFrame(data, columns=columns)
print("First 5 rows of the Historical MRI Appointment Dataset:")
print(historical_df.head())   

historical_df.to_csv('historical_mri_appointment_dataset_raw.csv', index=False)


