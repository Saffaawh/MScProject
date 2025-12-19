#this code will create a MRI Imaging Request CSV file
#It will use the patient details from the historical dataset and create a new dataset with imaging requests
#from the historical dataset it will take the patient number, age and gender 
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import sys
import logging
from sklearn.model_selection import train_test_split
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
scan_type_names = list(scan_types_details.keys())

# need to create some clinical indications for the scan types and probabilities in a dictionary mapping to the scan type above 
clinincal_mapping = { 'headache' : {'Brain': 0.8, 'Brain with Contrast': 0.2},
                        'neck pain' : {'Cervical Spine': 0.8, 'Cervical Spine with Contrast': 0.2},
                        'tumor' : {'Brain with Contrast': 0.4,  'Cervical Spine with Contrast': 0.1, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine with Contrast': 0.1, 'Whole Spine with Contrast': 0.3},
                        'infection' : {'Brain with Contrast': 0.4, 'Cervical Spine with Contrast': 0.2, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine with Contrast': 0.2, 'Whole Spine with Contrast': 0.1},
                        'dementia' : {'Brain': 0.9, 'Brain with Contrast': 0.1},
                        'aneurysm' : {'Brain with Contrast': 1.0, 'Brain': 0.9},
                        'stroke' : {'Brain': 0.7, 'Brain with Contrast': 0.3},
                        'multiple sclerosis': {'Brain': 0.5, 'Whole Spine': 0.5},
                        'back pain' : {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
                        'sciatica' : {'Lumbar Spine': 0.9, 'Lumbar Spine with Contrast': 0.1},
                        'spinal cord injury': {'Whole Spine': 0.4, 'Whole Spine with Contrast': 0.6},
                        'other' :{'Brain': 0.1, 'Brain with Contrast': 0.1, 'Cervical Spine': 0.1, 'Cervical Spine with Contrast': 0.1, 
                                 'Thoracic Spine': 0.1, 'Thoracic Spine with Contrast': 0.1, 'Lumbar Spine': 0.1, 
                                 'Lumbar Spine with Contrast': 0.1, 'Whole Spine': 0.1, 'Whole Spine with Contrast': 0.1}
                                 }
clinical_indications = list(clinincal_mapping.keys())
clinical_indication_probs = list(clinincal_mapping.values())
clinical_indication_cumprobs = np.cumsum(clinical_indication_probs)
# read in my historical dataset to get patient details
try:
    df_historical = pd.read_csv('final_historical_mri_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please ensure the dataset file is in the correct location.")
    exit()

    #use stratified sampling to get a representative sample of patients by class 
    num_samples = 100
    df_sampled, _ = train_test_split(df_historical[['Patient Number', 'Age', 'Gender', 'Class']], 
                                     train_size=num_samples, 
                                     stratify=df_historical['Class'], random_state=42)

#Typr of problem distribution 
type_of_problem = {'headache': 0.25, 
                   'neck pain': 0.15, 
                   'tumor': 0.1, 
                   'infection': 0.1, 
                   'dementia': 0.1, 
                   'aneurysm': 0.05, 
                   'stroke': 0.05, 
                   'multiple sclerosis': 0.05, 
                   'back pain': 0.1, 
                   'sciatica': 0.025,
                   'spinal cord injury': 0.025, 
                   'other': 0.05
                   }

type_of_problem_names = list(type_of_problem.keys())
type_of_problem_probs = list(type_of_problem.values()) 
type_of_problem_cumprobs = np.cumsum(type_of_problem_probs)
#function to select scan type based on clinical indication
def select_scan_type(clinical_indication):
    scan_type_probs = clinincal_mapping[clinical_indication]
    scan_types = list(scan_type_probs.keys())
    probs = list(scan_type_probs.values())
    cum_probs = np.cumsum(probs)
    rand_val = random.random()
    for i, cp in enumerate(cum_probs):
        if rand_val < cp:
            return scan_types[i]
    return scan_types[-1]  # Fallback in case of rounding errors   
# Create imaging requests
imaging_requests = []   
for index, row in df_sampled.iterrows():
    patient_number = row['Patient Number']
    age = row['Age']
    gender = row['Gender']
    #randomly assign a clinical indication based on the type_of_problem distribution        
    rand_val = random.random()
    for i, cp in enumerate(type_of_problem_cumprobs):
        if rand_val < cp:
            clinical_indication = type_of_problem_names[i]
            break
    #select a scan type based on the clinical indication
    scan_type = select_scan_type(clinical_indication)
    #create csv 
    class = row['Class']
    type_of_problem = clinical_indication