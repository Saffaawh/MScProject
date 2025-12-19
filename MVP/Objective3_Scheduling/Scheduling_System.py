#This code will schedule the new MRI appointments into available slots based on working hours, weekends, and holidays.
#It ensures no appointments are scheduled on Christmas Day and handles different working hours for weekdays, weekends, and bank holidays.
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, date
from workalendar.europe import UnitedKingdom
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Define scan types and their details
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
# need to define my MRI scanners I have 4 scanners with different rules 
# MRI Scanner 1 : 8am - 8pm Weekdays, 9am - 5pm Weekends and Bank Holidays - This scanner has a wide bore best fro claustrophobic patients
# MRI Scanner 2 : 8am - 8pm Weekdays, 9am - 5pm Weekends and Bank Holidays - This scanner does not have a removeable bed not suitable for poor mobility 
# MRI Scanner 3 : 8am - 8pm Weekdays, 9am - 5pm Weekends and Bank Holidays - This scanner has a very small bore not suitable for larger patients, claustrophobic patients
# MRI Scanner 4 : 8am - 8pm Weekdays, 9am - 5pm Weekends and Bank Holidays - This scanner is a research scanner only available when no research, only scans about 2 hours of outpatients a day, tight scehdule no transport patients
scanner_details = {
    'MRI Scanner 1': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': ['Wide Bore']},
    'MRI Scanner 2': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': []},
    'MRI Scanner 3': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': ['Small Bore']},
    'MRI Scanner 4': {'WeekdayHours': (8, 20), 'WeekendHours': (9, 17), 'SpecialFeatures': ['Research Only', 'Closed Wednesdays']}
}   


