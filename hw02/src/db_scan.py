import numpy.random
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque

from src.utils import *
import typing

from src.dsu import DisjointSetUnion


class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, leaf_size: int = 40, metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric

    def fit_predict(self, xs: np.ndarray):
        kd_tree = KDTree(xs, metric=self.metric, leaf_size=self.leaf_size)
        n_points = xs.shape[0]
        neighbours = kd_tree.query_radius(X=xs, r=self.eps)
        dsu = DisjointSetUnion(n_points)
        for i, neighs in enumerate(neighbours):
            if neighs.shape[0] < self.min_samples:
                continue
            for j in neighs:
                dsu.merge(i, j)

        cluster_id = {}
        current_cluster_id = 0
        for i in range(n_points):
            if i == dsu.find(i):
                cluster_id[i] = current_cluster_id
                current_cluster_id += 1

        return [cluster_id[dsu.find(i)] for i in range(n_points)]
