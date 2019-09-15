import random
import typing

import pandas
import numpy as np


def read_cancer_dataset(path_to_csv: str) -> typing.Tuple[typing.List, typing.List]:
    data = pandas.read_csv(path_to_csv).values.tolist()
    random.shuffle(data)
    return [row[1:] for row in data], [(1 if row[0] == 'M' else 0) for row in data]


def read_spam_dataset(path_to_csv: str) -> typing.Tuple[typing.List, typing.List]:
    data = pandas.read_csv(path_to_csv).values.tolist()
    random.shuffle(data)
    return [row[:-1] for row in data], [int(row[-1]) for row in data]


def train_test_split(
        xs: typing.List,
        ys: typing.List,
        ratio: float
) -> typing.Tuple[typing.List, typing.List, typing.List, typing.List]:
    assert len(xs) == len(ys)
    train_size = int(ratio * len(xs))
    return xs[:train_size], ys[:train_size], xs[train_size:], ys[train_size:]


def get_precision_recall_accuracy(y_pred: list, y_true: list) -> typing.Tuple[typing.List, typing.List, float]:
    assert len(y_pred) == len(y_true)
    n = len(y_pred)
    assert n > 0

    accuracy = sum(1 for yp, yt in zip(y_pred, y_true) if yp == yt) / n
    classes = set(y_true + y_pred)
    cn = len(classes)

    precision = [0.0] * cn
    recall = [0.0] * cn

    for i, y in enumerate(classes):
        tp = sum(1 for yp, yt in zip(y_pred, y_true) if yp == y and yt == y)  # true positive
        fp = sum(1 for yp, yt in zip(y_pred, y_true) if yp == y and yt != y)  # false positive
        fn = sum(1 for yp, yt in zip(y_pred, y_true) if yp != y and yt == y)  # false negative
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)

    return precision, recall, accuracy


def standard_scale(xs: np.ndarray) -> np.ndarray:
    mean = np.mean(xs, axis=0)
    std = np.std(xs, axis=0)
    return (xs - mean) / std


# scale [min, max] to [0, 1]
def zero_one_scale(xs: np.ndarray) -> np.ndarray
    _min = xs.min(axis=0)
    _max = xs.max(axis=0)
    return (xs - _min) / (_max - _min)


def max_abs_scale(xs: np.ndarray) -> np.ndarray:
    _max_abs = np.abs(xs).max(axis=0)
    return xs / _max_abs
