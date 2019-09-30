from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
from catboost import CatBoostClassifier
import typing

from src.random_forest_classifier import RandomForestClassifier


def gini(x: np.ndarray):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return np.sum(proba * (1 - proba))


def entropy(x: np.ndarray):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return -np.sum(proba * np.log2(proba))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion):
    y = np.concatenate((left_y, right_y))
    return criterion(y) - (criterion(left_y) * len(left_y) + criterion(right_y) * len(right_y)) / len(y)


def feature_importance(rfc: RandomForestClassifier):
    return np.sum(tree.out_of_bag_error() for tree in rfc.forest) / rfc.n_estimators


def most_important_features(importance, names, k: int = 20):
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]
