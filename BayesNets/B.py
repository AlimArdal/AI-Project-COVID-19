import pandas as pd

# Load data
data = pd.read_csv('COVID19_line_list_data.csv')

print(data['visiting Wuhan'])
print(data['symptom_onset'])

p_symptoms_given_wuhan = ((data['visiting Wuhan'] == 1) & (data['symptom_onset'].notnull())).mean()

print(f"The probability of being a true patient given that the person has symptoms and visited Wuhan is {p_symptoms_given_wuhan:.2f}")