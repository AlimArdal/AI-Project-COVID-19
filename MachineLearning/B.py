import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv('COVID19_line_list_data.csv')

# Convert age column to numeric and fill missing values with median age
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['age'] = data['age'].fillna(data['age'].median())

# One-hot encode categorical columns (gender and country)
data = pd.get_dummies(data, columns=['gender', 'country'])

# Select only numeric columns
data = data.select_dtypes(include='number')

# Select features (X) and target variable (y)
X = data.drop(['age'], axis=1)
y = data['age']

# Impute missing values with column mean
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and fit linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on test set
y_pred = reg.predict(X_test)

print(f"The predicted age based on gender and country is: {y_pred.mean():.2f} years old")

# Evaluate model performance using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
