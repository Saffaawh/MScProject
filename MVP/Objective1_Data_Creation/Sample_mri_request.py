#in this i want to take the MRI request with 5000 requests and take a smaple of 500 that represnts my patient population and flag distrubution 
import pandas as pd
from sklearn.model_selection import train_test_split
# Load the MRI request dataset
df_requests = pd.read_csv('mri_imaging_requests.csv')
# Display basic information about the dataset
print("Dataset loaded successfully.")
print(df_requests.info())
# Use stratified sampling to get a representative sample of patients by class
num_samples = 500
df_sampled, _ = train_test_split(df_requests[['Patient Number', 'Age', 'Gender', 'Class']], 
                                 train_size=num_samples,    
                                    stratify=df_requests['Class'], random_state=42)
# Save the sampled dataset to a new CSV file
df_sampled.to_csv('sampled_mri_requests_dataset.csv', index=False)
print("Sampled dataset created successfully with {} records.".format(len(df_sampled)))