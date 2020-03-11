                                                                                # GRADIENT DESCENT
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

plt.rcParams['figure.figsize'] = (8, 6)    # Figure ki width and height kitni hogi

data = pd.read_csv("KMeansData.csv")
print 'input data and shape'
print(data.shape)
print data.head()

# Getting the values and plot it
f1 = data['V1'].values
f2 = data['V2'].values

plt.scatter(f1, f2, c='black', s=10)
plt.show()

X = np.array(list(zip(f1, f2)))

# Euclidean Distance Calculate
def dist(a, b, ax=1):
    return np.linalg.norm(a-b, axis=ax)

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
print "C_x = ", C_x
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
print "C_y = ", C_y

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids (Random points) : ")
print(C)
print C.shape  #(3,2)

# Plotting along with the centroids
plt.scatter(f1, f2, s=10)
plt.scatter(C_x, C_y, marker='*', s=300, c='r')
plt.show()

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
print "C = \n", C
print "C_old \n", C_old

print 'len(X) = ', len(X)


# Cluster Labels(0, 1, 2)
clusters = np.zeros(len(X))

# Zero filled numpy array of 3000 elements
print "Clusters : ", clusters

# Error function --- Distance between new centroids and old centroids
error = dist(C, C_old, None)

# Loop will run till the error becomes zero
while error !=0:
    # Assigning each value to its closest cluster
     for i in range(len(X)):       # len(X)=3000
         distances = dist(X[i], C)    # distances = [12, 50, 9]
         cluster = np.argmin(distances)    # cluster = 2
         clusters[i] = cluster

     # Storing the old centroid values
     C_old = deepcopy(C)

     # Finding the new centroids by taking the average value
     for i in range(k):     # k=3 because we have to find 3 centroid locations
         points = [ X[j] for j in range(len(X)) if clusters[j]== i ]
         C[i] = np.mean(points, axis=0)
     error = dist(C, C_old, None)


colors = ['b', 'c', 'r']
fig, ax = plt.subplots()
for i in range(k):         # k=3
    points = np.array( [ X[j] for j in range(len(X)) if clusters[j] == i] )
    ax.scatter(points[:, 0], points[:, 1], s=25, c=colors[i] )

ax.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='y')

print 'Final Centroid : \n', C
plt.show()


#==============================================================================================
# scikit-learn
#==============================================================================================

from  sklearn.cluster import KMeans
print("KMeans of sklearn")
# Number of clusters
kmeans = KMeans(n_clusters=3)

data = pd.read_csv('KMeansData.csv')

f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))

kmeans = kmeans.fit(X)

# getting the cluster labels
labels = kmeans.predict(X)
print "labels: ", labels
# Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print "KMeans Algo Centroid values:: \n"
print "KMeans Manual centroids: \n", C
print "KMeans sklearn: centroids \n"
print(centroids)   # from scikit learn

