import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons
import scipy.special


class Module:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, d):
        raise NotImplementedError()

    def update(self, alpha):
        pass


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = np.random.uniform(-1, 1, size=(out_features, in_features + 1))

    def forward(self, X):
        x = X if len(X.shape) > 1 else X.reshape(-1, 1)
        x = np.vstack((x, np.ones((1, x.shape[1]))))
        return self.W.dot(x)

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        return np.vectorize(lambda t: max(t, 0))(x)

    def backward(self, d):
        pass


class Softmax(Module):
    def __init__(self):
        pass

    def forward(self, x):
        return scipy.special.softmax(x)

    def backward(self, d):
        pass
