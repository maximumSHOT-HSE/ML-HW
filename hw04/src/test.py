import numpy as np
import pandas
from numpy import random

from src.random_forest_classifier import RandomForestClassifier


def feature_importance(rfc: RandomForestClassifier):
    return sum(tree.out_of_bag_error() for tree in rfc.forest) / rfc.n_estimators


def most_important_features(importance, names, k: int = 20):
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=0)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y_age = [row[0] for row in dataset]
    y_sex = [row[1] for row in dataset]
    X = [row[2:] for row in dataset]

    return np.array(X), np.array(y_age), np.array(y_sex), list(dataframe.columns)[2:]


X, y = synthetic_dataset(1000)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)
print("Accuracy:", np.mean(rfc.predict(X) == y))
print("Importance:", feature_importance(rfc))
