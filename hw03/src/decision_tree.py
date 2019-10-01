import typing
import numpy as np


class DecisionTreeLeaf:
    def __init__(self, ys: np.ndarray):
        occur = dict()
        if ys.size > 0:
            for y in ys:
                if y in occur:
                    occur[y] += 1
                else:
                    occur[y] = 1
        for k in occur.keys():
            occur[k] /= ys.shape[0]
        self.probabilities = occur
        self.y = max(self.probabilities.keys(), key=lambda label: self.probabilities[label])
        self.size = ys.size


class DecisionTreeNode:
    def __init__(self, split_dim, split_value, left, right):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = left.size + right.size
