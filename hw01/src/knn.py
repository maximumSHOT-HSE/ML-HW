import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
import typing


def read_cancer_dataset(path_to_csv: str) -> typing.Tuple[typing.List, typing.List]:
    data = pandas.read_csv(path_to_csv).values.tolist()
    random.shuffle(data)
    return [row[1:] for row in data], [row[0] for row in data]


def read_spam_dataset(path_to_csv: str) -> typing.Tuple[typing.List, typing.List]:
    data = pandas.read_csv(path_to_csv).values.tolist()
    random.shuffle(data)
    return [row[:-1] for row in data], [int(row[-1]) for row in data]


def train_test_split(
        X: typing.List,
        y: typing.List,
        ratio: float
) -> typing.Tuple[typing.List, typing.List, typing.List, typing.List]:
    assert len(X) == len(y)
    train_size = int(ratio * len(X))
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


# X, y = read_cancer_dataset('resources/cancer.csv')
# X, y = read_spam_dataset('resources/spam.csv')
