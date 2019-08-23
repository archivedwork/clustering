##
#Compute clustering algorithm (e.g., k-means clustering) for different values of k.
# For instance, by varying k from 1 to 10 clusters.
# For each k, calculate the total within-cluster sum of square (wss).
# Plot the curve of wss according to the number of clusters k.
# The location of a bend (knee) in the plot is generally
# considered as an indicator of the appropriate number of clusters.
##

import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset= pd.read_csv('xclara.csv')
X = dataset.iloc[:, [0, 1]].values # get the column values

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append (kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()