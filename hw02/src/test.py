from sklearn.datasets import make_blobs, make_moons

from src.k_means import KMeans
from src.utils import *

if __name__ == '__main__':
    X_1, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
    # visualize_clasters(X_1, true_labels)
    X_2, true_labels = make_moons(400, noise=0.075)
    # visualize_clasters(X_2, true_labels)

    kmeans = KMeans(n_clusters=4, init='k-means++')
    kmeans.fit(X_1)
    labels = kmeans.predict(X_1)
    visualize_clasters(X_1, labels)
