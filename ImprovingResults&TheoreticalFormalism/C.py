import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

# Takes a lot of time to calculate (energy and time consuming)

data = pd.read_csv('COVID19_line_list_data.csv')

data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['age'] = data['age'].fillna(data['age'].median())
data['death'] = pd.to_numeric(data['death'], errors='coerce')
data['recovered'] = pd.to_numeric(data['recovered'], errors='coerce')

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svr = svm.SVR()

clf = GridSearchCV(svr, parameters)

X = data[['age', 'recovered', 'case_in_country']] 
y = data['death'] 

imp = SimpleImputer(strategy='mean')
X_imp = imp.fit_transform(X)

X_imp = X_imp[y.notnull()]
y = y[y.notnull()]

clf.fit(X_imp, y)

best_params = clf.best_params_
print(f"The best hyperparameters are: {best_params}")