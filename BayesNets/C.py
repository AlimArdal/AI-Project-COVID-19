import pandas as pd

# Load data
data = pd.read_csv('COVID19_line_list_data.csv')

data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['age'] = data['age'].fillna(data['age'].median())
data['death'] = pd.to_numeric(data['death'], errors='coerce')


# Calculate the probability of dying given that the person visited Wuhan
p_death_given_wuhan = (data['visiting Wuhan'] == 1) & (data['death'] == 1)
p_wuhan = (data['visiting Wuhan'] == 1).mean()
p_death_and_wuhan = p_death_given_wuhan.mean()
p_death_given_wuhan = p_death_and_wuhan / p_wuhan

print(f"The probability of dying given that the person visited Wuhan is {p_death_given_wuhan:.2f}")