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
from src.decision_tree import DecisionTree


class RandomForestClassifier:
    def __init__(
            self,
            criterion: str = 'gini',
            max_depth: int = None,
            min_samples_leaf: int = 1,
            max_features ='auto',
            n_estimators: int = 10
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.forest: typing.List[DecisionTree] = []

    def fit(self, xs: np.ndarray, ys: np.ndarray):
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
        return np.array([np.argmax(np.bincount(votes[:, i])) for i in range(xs.shape[1])])
