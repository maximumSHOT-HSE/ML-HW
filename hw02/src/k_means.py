from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import numpy as np
import numpy.random
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque

from src.utils import *
import typing


class KMeans:

    def __init__(self, n_clusters: int, init: str = 'random', max_iter: int = 300):
        assert n_clusters > 0
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None

    def sample_init(self, xs: np.ndarray):
        self.centroids = xs.copy()
        np.random.shuffle(self.centroids)
        self.centroids = self.centroids[:self.n_clusters]

    def random_init(self, shape: typing.Tuple):
        self.centroids = np.random.random(shape)[:self.n_clusters]

    def k_means_plus_plus_init(self, xs: np.ndarray):
        self.centroids = [xs[np.random.randint(0, xs.shape[0])]]
        for i in range(1, self.n_clusters):
            kd_tree = KDTree(np.array(self.centroids))
            dists, _ = kd_tree.query(X=xs)
            dists = np.array([d ** 2 for d in dists[:, 0]])

            next_d = np.random.uniform(0, dists.sum())
            next_c = 0
            cur_sum_d = 0
            while next_c < xs.shape[0] and cur_sum_d + dists[next_c] < next_d:
                cur_sum_d += dists[next_c]
                next_c += 1

            self.centroids.append(xs[next_c])

        self.centroids = np.array(self.centroids)

    def fit(self, xs: np.ndarray, y=None):
        if self.init == 'random':
            self.random_init(xs.shape)
        elif self.init == 'sample':
            self.sample_init(xs)
        elif self.init == 'k-means++':
            self.k_means_plus_plus_init(xs)
        else:
            raise Exception(f'Unrecognised init type: {self.init}')

    def predict(self, xs: np.ndarray):
        kd_tree = KDTree(self.centroids)
        _, i = kd_tree.query(xs)
        result = i[:, 0]
        if np.unique(result).shape[0] != self.n_clusters:
            return self.predict(xs)
        else:
            return result
