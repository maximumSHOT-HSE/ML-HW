from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import typing
import math


def calc_probabilities(labels: np.ndarray) -> np.ndarray:
    labels = labels.reshape(labels.size)
    n_labels = labels.shape[0]
    occur = dict()
    for label in labels:
        if label in occur:
            occur[label] += 1
        else:
            occur[label] = 1
    return np.array([occ / n_labels for occ in occur.values()])


def gini(labels: np.ndarray):
    return sum(p * (1 - p) for p in calc_probabilities(labels))


def entropy(labels: np.ndarray):
    return -sum(p * math.log(p) for p in calc_probabilities(labels))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion):
    left_size = left_y.size
    right_size = right_y.size
    return (left_size + right_size) * criterion(np.concatenate((left_y, right_y))) - \
           (left_size * criterion(left_y)) - \
           (right_size * criterion(right_y))
