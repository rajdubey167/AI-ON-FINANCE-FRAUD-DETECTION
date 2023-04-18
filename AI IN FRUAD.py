import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load the dataset
df = pd.read_csv('finance_data.csv')
# Remove irrelevant columns
df.drop(['ID', 'Date'], axis=1, inplace=True)
# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# Cluster the data using KMeans
kmeans = KMeans(n_clusters=2, random_state=42).fit(df_scaled)
# Assign labels to original dataset
df['Cluster'] = kmeans.labels_
# Identify potential fraudsters in each cluster
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
fraudsters = []
for i in range(2):
    center = cluster_centers[i]
    mask = np.all(df[df['Cluster']==i].drop('Cluster', axis=1) > center,axis=1)
    fraudsters += df[df['Cluster']==i][mask].index.tolist()
print(fraudsters)