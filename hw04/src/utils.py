import numpy as np


def gini(x: np.ndarray):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return sum(proba * (1 - proba))


def entropy(x: np.ndarray):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return -sum(proba * np.log2(proba))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion):
    y = np.concatenate((left_y, right_y))
    return criterion(y) - (criterion(left_y) * len(left_y) + criterion(right_y) * len(right_y)) / len(y)
