import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
import typing


class Point:

    def __init__(self, vector, id):
        self.vector = vector
        self.id = id

    def get_dist(self, o):
        return np.linalg.norm(self.vector - o.vector)


class Node:
    def __init__(
            self,
            axis: int,
            axis_value,
            points_number: int,
            left_son=None,
            right_son=None,
            points: typing.List[Point] = None
    ):
        self.axis = axis
        self.axis_value = axis_value
        self.points_number = points_number
        self.left_son = left_son
        self.right_son = right_son
        self.points = points


class KDTree:

    def build_tree(self, points, axis, dimension, leaf_size):
        pivot = np.median(np.array([p.vector[axis] for p in points]))
        points_number = len(points)

        if points_number <= leaf_size:
            return Node(axis, pivot, points_number, points=points)

        left_part = [p for p in points if p.vector[axis] < pivot]
        right_part = [p for p in points if p.vector[axis] >= pivot]

        left_son = self.build_tree(left_part, (axis + 1) % dimension, dimension, leaf_size)
        right_son = self.build_tree(right_part, (axis + 1) % dimension, dimension, leaf_size)

        return Node(axis, pivot, points_number, left_son, right_son, points)

    def find_circle_boundary(self, point, node, k):
        node_to = node.left_son if point[node.axis] < node.axis_value else node.right_son
        if node_to and node_to.points_number >= k:
            return self.find_circle_boundary(point, node_to, k)
        node.points_number.sort(key=lambda p: point.get_dist(p))
        return point.get_dist(node.points[k - 1])

    def find_knn(self, point, k):
        assert self.root.points_number >= k
        radius = self.find_circle_boundary(point, self.root, k)

    def __init__(self, points, leaf_size=40):
        points = [Point(vector, i) for i, vector in enumerate(points)]
        points_number, dimension = len(points)
        assert points_number > 0
        assert dimension > 0
        self.root = self.build_tree(points, 0, dimension, leaf_size)

    def query(self, xs, k=1, return_distance=True):
        if self.root.points_number >= k:
            result = [self.find_knn(x, k) for x in xs]
        else:
            result = [self.root.points for _ in xs]
