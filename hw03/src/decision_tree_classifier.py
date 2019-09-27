from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import typing
import math

from src.utils import *
from src.decision_tree import *


class DecisionTreeClassifier:

    def __init__(self, criterion: str = "gini", max_depth=None, min_samples_leaf: int = 1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = gini if criterion == "gini" else entropy

    def build_tree(self, xs: np.ndarray, ys: np.ndarray, depth: int = 0):
        if np.sort(np.unique(ys)).size == 1 or depth == self.max_depth:
            return DecisionTreeLeaf(ys)

        best_dim = -1
        best_separator = -1
        best_ig = 0
        ids = [i for i in range(xs.shape[0])]

        for dim in range(xs.shape[1]):
            ids.sort(key=lambda i: xs[i][dim])
            for j in range(xs.shape[0]):
                if j + 1 < xs.shape[0] and xs[ids[j]][dim] == xs[ids[j + 1]][dim] or \
                        min(j + 1, xs.shape[0] - j - 1) < self.min_samples_leaf:
                    continue
                left_y = np.array([ys[ids[q]] for q in range(j + 1)])
                right_y = np.array([ys[ids[q]] for q in range(j + 1, xs.shape[0])])
                ig = gain(left_y, right_y, self.criterion)

                if best_dim == -1 or ig > best_ig:
                    best_ig = ig
                    best_dim = dim
                    best_separator = j

        if best_dim == -1:
            return DecisionTreeLeaf(ys)

        ids.sort(key=lambda i: xs[i][best_dim])

        left_xs = np.array([xs[ids[q]] for q in range(best_separator + 1)])
        left_ys = np.array([ys[ids[q]] for q in range(best_separator + 1)])

        right_xs = np.array([xs[ids[q]] for q in range(best_separator + 1, xs.shape[0])])
        right_ys = np.array([ys[ids[q]] for q in range(best_separator + 1, xs.shape[0])])

        left_son = self.build_tree(left_xs, left_ys, depth + 1)
        right_son = self.build_tree(right_xs, right_ys, depth + 1)

        return DecisionTreeNode(best_dim, xs[best_separator + 1][best_dim], left_son, right_son)

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        ys = ys.reshape(ys.size)
        self.root = self.build_tree(xs, ys)

    def get_probabilities(self, x: np.ndarray, node) -> dict:
        if isinstance(node, DecisionTreeLeaf):
            return node.probabilities
        if x[node.split_dim] < node.split_value:
            return self.get_probabilities(x, node.left)
        else:
            return self.get_probabilities(x, node.right)

    def predict_probabilities(self, xs: np.ndarray) -> typing.List[dict]:
        return [self.get_probabilities(x, self.root) for x in xs]

    def predict(self, xs: np.ndarray) -> np.ndarray:
        probabilities = self.predict_probabilities(xs)
        return np.array([max(p.keys(), key=lambda k: p[k]) for p in probabilities])

    def explain(self, x: np.ndarray, node):
        if isinstance(node, DecisionTreeLeaf):
            return node.y, ''
        if x[node.split_dim] < node.split_value:
            y, s = self.explain(x, node.left)
            s += f'\nx[{node.split_dim}] < {node.split_value}'
            return y, s
        else:
            y, s = self.explain(x, node.right)
            s += f'\nx[{node.split_dim}] >= {node.split_dim}'
            return y, s


def predict_explain(dtc: DecisionTreeClassifier, xs: np.ndarray) -> np.ndarray:
    return np.array([dtc.explain(x, dtc.root) for x in xs])
