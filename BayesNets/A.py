import pandas as pd

data = pd.read_csv('COVID19_line_list_data.csv')

# Calculate the probability of having symptoms given that the person visited Wuhan
p_symptoms_given_wuhan = (data['visiting Wuhan'] == 1) & (data['symptom_onset'].notnull())
p_wuhan = (data['visiting Wuhan'] == 1).mean()
p_symptoms_and_wuhan = p_symptoms_given_wuhan.mean()
p_symptoms_given_wuhan = p_symptoms_and_wuhan / p_wuhan

print(f"The probability of having symptoms given that the person visited Wuhan is {p_symptoms_given_wuhan:.2f}")