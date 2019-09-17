import numpy.random
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque

from src.utils import *
import typing


class DisjointSetUnion:

    def __init__(self, n):
        self.n = n
        self.parent = [i for i in range(n)]

    def find(self, x):
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        self.parent[x] = y
