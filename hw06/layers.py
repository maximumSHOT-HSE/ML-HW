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
        self.x = None
        self.d = None

    def forward(self, X):
        self.x = copy.deepcopy(X)
        self.x = self.x if len(self.x.shape) > 1 else self.x.reshape(-1, 1)
        self.x = np.vstack((self.x, np.ones((1, self.x.shape[1]))))
        return self.W.dot(self.x)

    def backward(self, d):
        self.d = copy.deepcopy(d).reshape(1, -1)
        pd = self.d.dot(self.W)
        return pd.reshape(-1)[:-1]

    def update(self, alpha):
        self.W -= alpha * (np.sum(self.x, axis=1).reshape(-1, 1).dot(self.d)).T


class ReLU(Module):

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = copy.deepcopy(x)
        self.x = self.x if len(self.x.shape) > 1 else self.x.reshape(-1, 1)
        self.y = np.vectorize(lambda t: max(t, 0))(self.x)
        return self.y

    def backward(self, d):
        return np.vectorize(lambda t: 1 if t >= 0 else 0)(d)


class Softmax(Module):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = copy.deepcopy(x)
        self.x = self.x if len(self.x.shape) > 1 else self.x.reshape(-1, 1)
        self.y = scipy.special.softmax(x, axis=0)
        return copy.deepcopy(self.y)

    def backward(self, d):
        c = copy.deepcopy(d).reshape(1, -1)
        c = -c.dot(self.y)
        c = np.repeat(c, len(self.y))
        c = c * self.y
        return np.sum(c, axis=1) * d
