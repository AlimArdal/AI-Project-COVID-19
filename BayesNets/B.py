import pandas as pd

# Load data
data = pd.read_csv('COVID19_line_list_data.csv')

print(data['visiting Wuhan'])
print(data['symptom_onset'])
# Calculate the probability of being a true patient given that the person has symptoms and visited Wuhan
p_patient_given_symptoms_wuhan = (data['visiting Wuhan'] == 1) & (data['symptom_onset'].notnull())
p_symptoms_wuhan = ((data['visiting Wuhan'] == 1) & (data['symptom_onset'].notnull())).mean()
p_patient_and_symptoms_wuhan = p_patient_given_symptoms_wuhan.mean()
p_patient_given_symptoms_wuhan = p_patient_and_symptoms_wuhan / p_symptoms_wuhan

print(f"The probability of being a true patient given that the person has symptoms and visited Wuhan is {p_patient_given_symptoms_wuhan:.2f}")