import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

np.random.seed(0)

# Generate first
x1, _ = make_circles(n_samples=1000, factor=0.99, noise=0.1)

x1 = x1 + (1, 1)

x2, _ = make_circles(n_samples=1000, factor=0.99, noise=0.05)

x2 = x2 + (-0.2, 0)

x = np.concatenate([x1, x2])

# run kmedoids
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=2, random_state=0).fit(x)

plt.scatter(x[:, 0], x[:, 1], alpha=0.5, s=4)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], c='yellow', s=100, label='KMedoids',
            marker='x')

# run kmeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(x)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, label='KMeans', marker='x')
# keep aspect ratio
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title('KMeans vs KMedoids example (K=2)')
plt.xlabel('X')
plt.ylabel('Y')
# plt.show()
plt.savefig('output/kmeans-vs-kmedoids.pdf')
