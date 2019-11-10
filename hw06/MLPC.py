import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons
import scipy.special
from layers import *
import layers
import typing


class MLPClassifier:

    def __init__(self, modules: typing.List[Module], epochs=40, alpha=0.01):
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha

    def epoch(self):
        curx = self.X
        for layer in self.modules:
            curx = layer.forward(curx)
        curx = -1 / curx
        curd = np.array([sum(curx[i, y]) for i in range(len(curx))])
        for layer in reversed(self.modules):
            curd = layer.backward(curd)
            layer.update(self.alpha)

    def fit(self, X, y):
        self.X = copy.deepcopy(X).T
        self.y = copy.deepcopy(np.array(y)).reshape(-1)
        for _ in range(self.epochs):
            self.epoch()

    def predict_proba(self, X):
        curx = copy.deepcopy(X).T
        for layer in self.modules:
            curx = layer.forward(curx)
        return curx.T

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)
