import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
import typing


def read_cancer_dataset(path_to_csv):
    data = pandas.read_csv(path_to_csv).values.tolist()
    random.shuffle(data)
    return [row[1:] for row in data], [row[0] for row in data]


def read_spam_dataset(path_to_csv):
    data = pandas.read_csv(path_to_csv).values.tolist()
    random.shuffle(data)
    return [row[:-1] for row in data], [int(row[-1]) for row in data]


# X, y = read_cancer_dataset('resources/cancer.csv')
# X, y = read_spam_dataset('resources/spam.csv')


