import typing

import math
import numpy as np

from src.decision_tree import DecisionTree


def majority(a: np.ndarray):
    count = dict()
    for x in a:
        if x in count:
            count[x] += 1
        else:
            count[x] = 1
    best_key = -1
    best_value = -1
    for key, value in count.items():
        if value > best_value:
            best_value = value
            best_key = key
    return best_key


class RandomForestClassifier:
    def __init__(
            self,
            criterion: str = 'gini',
            max_depth: int = None,
            min_samples_leaf: int = 1,
            max_features='auto',
            n_estimators: int = 10
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.forest: typing.List[DecisionTree] = []

        self.xs = None
        self.ys = None

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = xs
        self.ys = ys
        self.forest.clear()
        for _ in range(self.n_estimators):
            self.forest.append(
                DecisionTree(
                    xs,
                    ys,
                    self.criterion,
                    self.max_depth,
                    self.min_samples_leaf,
                    self.max_features
                )
            )

    def predict(self, xs: np.ndarray) -> np.ndarray:
        votes = np.array([tree.predict(xs) for tree in self.forest])
        return np.array([majority(votes[:, i]) for i in range(votes.shape[1])])
