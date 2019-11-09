import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons
import scipy.special
from layers import Linear

linear = Linear(10, 5)

X1 = np.arange(10)

print(linear.forward(X1))

X2 = np.arange(90).reshape(10, 9)

print(linear.forward(X2).shape)
