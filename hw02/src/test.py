from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
import typing

from src.utils import *
from src.k_means import KMeans


if __name__ == '__main__':
    # image = read_image("../resources/image.jpg")
    # result = clusterize_image(image)
    # show_image(result)
    # save_image(result, "../resources/result_without_shift.jpg")

    # X_1, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
    # # visualize_clasters(X_1, true_labels)
    # X_2, true_labels = make_moons(400, noise=0.075)
    # # visualize_clasters(X_2, true_labels)
    #
    # kmeans = KMeans(n_clusters=4, init='k-means++')
    # kmeans.fit(X_1)
    # labels = kmeans.predict(X_1)
    # visualize_clasters(X_1, labels)

    image = read_image("../resources/image.jpg")
    result = clusterize_image(image)
    show_image(result)
    save_image(result, "result")
