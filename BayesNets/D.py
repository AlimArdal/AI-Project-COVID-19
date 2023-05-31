import pandas as pd

# Load data
data = pd.read_csv('COVID19_line_list_data.csv')

data['symptom_onset'] = pd.to_datetime(data['symptom_onset'], format='%m/%d/%Y', errors='coerce')
data['recovered'] = pd.to_datetime(data['recovered'], format='%m/%d/%Y', errors='coerce')

# Remove rows with incorrect dates in the recovered column
data = data[data['recovered'] != '12-30-1899']

# Calculate recovery interval
data['recovery_interval'] = (data['recovered'] - data['symptom_onset']).dt.days

# Convert dates to strings in desired format
data['symptom_onset'] = data['symptom_onset'].dt.strftime('%m-%d-%Y')
data['recovered'] = data['recovered'].dt.strftime('%m-%d-%Y')

# Calculate average recovery interval for patients who visited Wuhan
avg_recovery_interval = data.loc[data['visiting Wuhan'] == 1, 'recovery_interval'].mean()

print(f"The average recovery interval for patients who visited Wuhan is {avg_recovery_interval:.2f} days")
