import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv('COVID19_line_list_data.csv')

# One-hot encode gender column
data = pd.get_dummies(data, columns=['gender'])

# Select features and target variable
X = data[['age', 'gender_female', 'gender_male', 'visiting Wuhan', 'from Wuhan']]
y = data['death']

# Impute missing values with column mean
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and fit K-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on test set
y_pred = knn.predict(X_test)

# Evaluate model performance
confusion = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print(f"Confusion matrix:\n{confusion}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
