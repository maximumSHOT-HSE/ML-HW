from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
from catboost import CatBoostClassifier
import typing
import math

from src.utils import *


def out_of_bag(size, ids: np.ndarray) -> np.ndarray:
    used = set()
    for i in ids:
        used.add(i)
    return np.array([i not in used for i in range(size)])


def bagging(size: int) -> np.ndarray:
    ids = []
    for _ in range(size):
        ids.append(np.random.randint(size))
    return np.array(ids)


class DecisionTreeLeaf:
    def __init__(self, ys: np.ndarray):
        occur = dict()
        if ys.size > 0:
            for y in ys:
                if y in occur:
                    occur[y] += 1
                else:
                    occur[y] = 1
        for k in occur.keys():
            occur[k] /= ys.shape[0]
        self.probabilities = occur
        self.y = max(self.probabilities.keys(), key=lambda label: self.probabilities[label])
        self.size = ys.size


class DecisionTreeNode:
    def __init__(self, split_dim, split_value, left, right):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = left.size + right.size


class DecisionTree:
    def __init__(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            criterion: str = 'gini',
            max_depth=None,
            min_samples_leaf: int = 1,
            max_features='auto'
    ):
        self.criterion = gini if criterion == 'gini' else entropy
        self.max_depth = max_depth if max_depth else math.inf
        self.min_samples_leaf = min_samples_leaf
        self.max_feature = min(int(max_features), xs.shape[1]) if max_features != 'auto' \
            else math.ceil(xs.shape[1] ** 0.5)
        self.xs = xs
        self.ys = ys
        self.bag = bagging(len(xs))
        self.out_of_bag = out_of_bag(len(xs), self.bag)
        self.root = self.build_tree(self.xs[self.bag], self.ys[self.bag])

    def build_tree(self, xs: np.ndarray, ys: np.ndarray, depth: int = 0):
        if np.sort(np.unique(ys)).size == 1 or depth == self.max_depth:
            return DecisionTreeLeaf(ys)

        best_dim = -1
        best_separator = -1
        best_ig = 0
        ids = [i for i in range(xs.shape[0])]

        dims = np.array([i for i in range(xs.shape[1])])
        np.random.shuffle(dims)
        dims = dims[:self.max_feature]

        for dim in dims:

            left_y = np.array([ys[ids[q]] for q in ids if ys[ids[q]] == 0])
            right_y = np.array([ys[ids[q]] for q in ids if ys[ids[q]] == 1])

            if left_y.shape[0] < self.min_samples_leaf or right_y.shape[0] < self.min_samples_leaf:
                continue

            ig = gain(left_y, right_y, self.criterion)

            if best_dim == -1 or ig > best_ig:
                best_ig = ig
                best_dim = dim

        if best_dim == -1:
            return DecisionTreeLeaf(ys)

        left_xs = np.array([xs[ids[q]][best_dim] for q in ids if ys[ids[q]] == 0])
        left_ys = np.array([ys[ids[q]] for q in ids if ys[ids[q]] == 0])

        right_xs = np.array([xs[ids[q]][best_dim] for q in ids if ys[ids[q]] == 1])
        right_ys = np.array([ys[ids[q]] for q in ids if ys[ids[q]] == 1])

        left_son = self.build_tree(left_xs, left_ys, depth + 1)
        right_son = self.build_tree(right_xs, right_ys, depth + 1)

        return DecisionTreeNode(best_dim, best_separator, left_son, right_son)

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

    def out_of_bag_error(self):
        xs = self.xs[self.out_of_bag]
        ys = self.ys[self.out_of_bag]
        y_pred = self.predict(xs)
        err = sum(1 for y, i in enumerate(ys) if y != y_pred[i])
        errors = []
        for j in range(xs.shape[1]):
            xsj = xs.copy()
            np.random.shuffle(xsj[:, j])
            y_pred = self.predict(xsj)
            err_j = sum(1 for y, i in enumerate(ys) if y != y_pred[i])
            errors.append(err_j - err)
        return np.array(errors)
