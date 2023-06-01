import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def parse_date(date_string):
    try:
        return pd.to_datetime(date_string, format='%m/%d/%Y')
    except ValueError:
        try:
            return pd.to_datetime(date_string, format='%d-%m-%Y')
        except ValueError:
            return pd.NaT

# COVID19_line_list_data.csv
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
# Charger les données à partir d'un fichier CSV
data = pd.read_csv('COVID19_line_list_data.csv')
print(data)
print("----------------------------------------------")

# DATES --------------------------------------------------------------------------------------------------------------------------------------------------------

def is_bad_date(date_string):
    return False
    
data['reporting date'] = data['reporting date'].apply(parse_date)
data['reporting date'] = data['reporting date'].dt.strftime('%d/%m/%Y')
data['reporting date'] = pd.to_datetime(data['reporting date'], format='%d/%m/%Y')
mean_date = data['reporting date'].mean()
data['reporting date'] = data['reporting date'].fillna(mean_date)
data['reporting date'] = data['reporting date'].apply(lambda x: x.toordinal())
print(data['reporting date'])

data['symptom_onset'] = data['symptom_onset'].apply(parse_date)
data['symptom_onset'] = data['symptom_onset'].dt.strftime('%d/%m/%Y')
data['symptom_onset'] = pd.to_datetime(data['symptom_onset'], format='%d/%m/%Y')
median_date = data['symptom_onset'].dropna().median()
data['symptom_onset'] = data['symptom_onset'].fillna(median_date)
data['symptom_onset'] = data['symptom_onset'].apply(lambda x: x.toordinal())

data['hosp_visit_date'] = data['hosp_visit_date'].apply(parse_date)
data['hosp_visit_date'] = data['hosp_visit_date'].dt.strftime('%d/%m/%Y')
data['hosp_visit_date'] = pd.to_datetime(data['hosp_visit_date'], format='%d/%m/%Y', errors='coerce')
median_date2 = data['hosp_visit_date'].dropna().median()
data['hosp_visit_date'] = data['hosp_visit_date'].fillna(median_date2)
data['hosp_visit_date'] = data['hosp_visit_date'].apply(lambda x: x.toordinal())

data['exposure_start'] = data['exposure_start'].apply(parse_date)
data['exposure_start'] = data['exposure_start'].dt.strftime('%d/%m/%Y')
data['exposure_start'] = pd.to_datetime(data['exposure_start'], format='%d/%m/%Y', errors='coerce')
median_date3 = data['exposure_start'].dropna().median()
data['exposure_start'] = data['exposure_start'].fillna(median_date3)
data['exposure_start'] = data['exposure_start'].apply(lambda x: x.toordinal())

data['exposure_end'] = data['exposure_end'].apply(parse_date)
data['exposure_end'] = data['exposure_end'].dt.strftime('%d/%m/%Y')
data['exposure_end'] = pd.to_datetime(data['exposure_end'], format='%d/%m/%Y', errors='coerce')
median_date4 = data['exposure_end'].dropna().median()
data['exposure_end'] = data['exposure_end'].fillna(median_date4)
data['exposure_end'] = data['exposure_end'].apply(lambda x: x.toordinal())

# DATES --------------------------------------------------------------------------------------------------------------------------------------------------------



# Remplacer les valeurs non numériques dans la colonne 'age' et 'death' par la médiane de cette colonne
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['age'] = data['age'].fillna(data['age'].median())

data['death'] = pd.to_numeric(data['death'], errors='coerce')
data['recovered'] = pd.to_numeric(data['recovered'], errors='coerce')
data['case_in_country'] = data['case_in_country'].fillna(0)

# Supprimer les colonnes sans nom
data = data.filter(regex='[^Unnamed]')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

print(data)
print("----------------------------------------------")


# Remplacer les valeurs manquantes par la moyenne de la variable concernée lorsque cela est possible
data = data.apply(lambda x: x.fillna(x.mean()) if np.issubdtype(x.dtype, np.number) else x)
print(data)
print("----------------------------------------------")

print(data.dtypes)

print("----------------------------------------------")

# Calculer la corrélation entre les variables 'age' et 'death'
corr = data['age'].corr(data['death'])
print(f"La corrélation entre l'âge et la mort est de {corr:.2f}")
print("----------------------------------------------")


corr = data[['age','recovered','visiting Wuhan','from Wuhan','case_in_country','If_onset_approximated','death','reporting date','symptom_onset','hosp_visit_date','exposure_start','exposure_end']].corr().apply(lambda x: round(x, 2))

print(corr)
print("----------------------------------------------")


# Calculer la matrice de corrélation
'''corr_matrix = data.corr()
print(corr_matrix)'''



# BayesNet B -------------------------------------------------------------------------------------------------------------

# Calculate the probability of being a true patient given that the person has symptoms and visited Wuhan
p_patient_given_symptoms_wuhan = (data['visiting Wuhan'] == 1) & (data['symptom_onset'].notnull())
p_symptoms_wuhan = ((data['visiting Wuhan'] == 1) & (data['symptom_onset'].notnull())).mean()
p_patient_and_symptoms_wuhan = p_patient_given_symptoms_wuhan.mean()
p_patient_given_symptoms_wuhan = p_patient_and_symptoms_wuhan / p_symptoms_wuhan

print(f"The probability of being a true patient given that the person has symptoms and visited Wuhan is {p_patient_given_symptoms_wuhan:.2f}")

# BayesNet C -------------------------------------------------------------------------------------------------------------
# Calculate the probability of dying given that the person visited Wuhan
p_death_given_wuhan = (data['visiting Wuhan'] == 1) & (data['death'] == 1)
p_wuhan = (data['visiting Wuhan'] == 1).mean()
p_death_and_wuhan = p_death_given_wuhan.mean()
p_death_given_wuhan = p_death_and_wuhan / p_wuhan

print(f"The probability of dying given that the person visited Wuhan is {p_death_given_wuhan:.2f}")



# Standardize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(corr)

# Perform PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_std)

# Plot scatter plot
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.show()