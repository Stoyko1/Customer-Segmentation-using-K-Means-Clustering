# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 18:30:59 2023

@author: S
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

file_path = 'Customers Dataset.csv'
customers_data = pd.read_csv(file_path)

missing_values = customers_data.isnull().sum()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(customers_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])


kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
customers_data['Cluster'] = clusters


plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[clusters == 0, 0], scaled_features[clusters == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(scaled_features[clusters == 1, 0], scaled_features[clusters == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(scaled_features[clusters == 2, 0], scaled_features[clusters == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(scaled_features[clusters == 3, 0], scaled_features[clusters == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(scaled_features[clusters == 4, 0], scaled_features[clusters == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()


print(customers_data.head())


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='Gender', data=customers_data, palette='bright')
plt.title('Age vs Annual Income by Gender')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Gender', data=customers_data, palette='bright')
plt.title('Age vs Spending Score by Gender')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=customers_data, palette='bright')
plt.title('Annual Income vs Spending Score by Gender')
plt.show()
