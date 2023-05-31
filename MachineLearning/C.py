import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv('COVID19_line_list_data.csv')

# One-hot encode gender column
data = pd.get_dummies(data, columns=['gender'])

# Select features
X = data[['age', 'gender_female', 'gender_male', 'visiting Wuhan', 'from Wuhan']]
y = data['age']

# Impute missing values with column mean
imputer = SimpleImputer()
X = imputer.fit_transform(X)
y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Determine optimal number of clusters using silhouette score
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_n_clusters = 2 + silhouette_scores.index(max(silhouette_scores))

# Apply K-means clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Plot boxplots of age distribution in each cluster with y-axis label
data['cluster'] = cluster_labels
data.boxplot(column='age', by='cluster')
plt.ylabel('Age')
plt.show()
