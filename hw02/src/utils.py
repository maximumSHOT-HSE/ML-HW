import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.k_means import KMeans


def visualize_clasters(X, labels):
    unique_labels = np.unique(labels)
    unique_colors = np.random.random((len(unique_labels), 3))
    colors = [unique_colors[l] for l in labels]
    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


def clusters_statistics(flatten_image, cluster_colors, cluster_labels):
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    for remove_color in range(3):
        axes_pair = axes[remove_color]
        first_color = 0 if remove_color != 0 else 2
        second_color = 1 if remove_color != 1 else 2
        axes_pair[0].scatter(
            [p[first_color] for p in flatten_image],
            [p[second_color] for p in flatten_image],
            c=flatten_image,
            marker='.'
        )
        axes_pair[1].scatter(
            [p[first_color] for p in flatten_image],
            [p[second_color] for p in flatten_image],
            c=[cluster_colors[c] for c in cluster_labels],
            marker='.'
        )
        for a in axes_pair:
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
    plt.show()


def read_image(path: str):
    return np.flip(cv2.imread(path), 2)


def show_image(image):
    plt.figure(figsize=np.array(image.shape[:-1]) / 50)
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_image(image, path):
    cv2.imwrite(path, np.flip(image, 2))


def clusterize_image(image):
    height, width, color_bytes = image.shape

    xs = image.reshape((-1, color_bytes))
    _min = xs.min(axis=0)
    _max = xs.max(axis=0)
    xs = (xs - _min) / (_max - _min)

    kmeans = KMeans(n_clusters=64, init='k-means++', max_iter=300)
    kmeans.fit(xs)

    clusters = kmeans.predict(xs)
    cluster_colors = kmeans.centroids
    clusters_statistics(xs, cluster_colors, clusters)  # Very slow (:

    cluster_colors = (cluster_colors * (_max - _min) + _min).astype(int)
    recolored = np.array([cluster_colors[c] for c in clusters]).reshape((height, width, color_bytes))

    return recolored
