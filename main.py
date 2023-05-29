import pandas as pd
import numpy as np
#import seaborn as sns; sns.set()


#a=sns.distplot(data)

# COVID19_line_list_data.csv
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
# Charger les données à partir d'un fichier CSV
data = pd.read_csv('COVID19_line_list_data.csv')
print(data)
print("----------------------------------------------")


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


# Supprimer les lignes contenant des valeurs manquantes
data = data.dropna()
print(data)
print("----------------------------------------------")

data2=data.values().astype(str)
print (data2)

# Calculer la matrice de corrélation
corr_matrix = data2.corr()
print(corr_matrix)

# Afficher les corrélations avec la variable cible
target_variable = 'age'
print(corr_matrix[target_variable].sort_values(ascending=False))
print("----------------------------------------------")