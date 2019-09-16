from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque


class DisjointSetUnion:

    def __init__(self, n_vertices: int):
        self.n_vertices = n_vertices
        self.parent = [i for i in range(0, n_vertices)]
        self.size = [1] * n_vertices

    def find(self, x: int):
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x: int, y: int):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.parent[y] = x
            self.size[x] += self.size[y]
