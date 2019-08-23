from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# importing the datasets
data = pd.read_csv('xclara.csv')
print(data.shape)
data.head()

# getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
#plt.scatter(f1, f2, c='black', s=7)


#Euclidean distance calculator
def dist(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)

#number of clusters
k = 3

# X coordinate of random centroids
C_x = np.random.randint(0, np.max(X) - 20, size=k)

# Y coordinate of random centroids
C_y = np.random.randint(0, np.max(X) - 20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)

#plotting along with centroids
#plt.scatter(C_x, C_y, marker='*', s=200, c='g')

#To store the value of centroids when it updates
C_old = np.zeros(C.shape)

#Cluster Labels(0,1,2)
clusters = np.zeros(len(X))

#Error func. - Dist between new centroids and old centroids
error = dist(C, C_old, None)

# Loop until the error becomes zero
while error != 0:
		# assigning each value to its closest cluster
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			clusters[i] = cluster
		# Storing the old centroid values
		C_old = deepcopy(C)
		# Finding the new centroids by taking average value
		for i in range(k):
			points = [X[j] for j in range(len(X)) if clusters[j] == i]
			C[i] = np.mean(points, axis=0)
		error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
	points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
	ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='b')




# The scikit-learn approach
# Example 1
# We will use the same dataset in this example.
from sklearn.cluster import KMeans

# number of clusters
kmeans = KMeans(n_clusters=3)
#Fitting the input data
kmeans = kmeans.fit(X)
#Getting the cluster labels
labels = kmeans.predict(X)
#Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print(C) # from scratch
print(centroids) #from sci-kit learn
# You can see that the centroid values are equal, but in different order.


# Example 2
# We will generate a new dataset using make_blobs function
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize'] = (16, 9)

# creating a sample random  dataset with 4 clusters and plot them
X, y = make_blobs(n_samples=800, n_features=3, centers=4)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])

# Initializing KMeans
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='y')
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='b', s=1000)


plt.show()


