import typing

import numpy as np


def get_dist(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a - b)


class Point:

    def __init__(self, vector, i):
        self.vector = vector
        self.id = i

    def get_dist(self, o: np.ndarray):
        return get_dist(self.vector, o)


class Node:

    def __init__(
            self,
            axis: int,
            axis_value: float,
            points_number: int,
            left_son=None,
            right_son=None,
            points: typing.List[Point] = None
    ):
        self.axis = axis
        self.axis_value = axis_value
        self.points_number = points_number
        self.points = points
        self.left_son = left_son
        self.right_son = right_son

        self.box_center = np.array([])
        self.box_radius = 0

        self.build_boundaries()

    def build_boundaries(self):
        up_boundary = np.array([p.vector for p in self.points]).max(axis=0)
        down_boundary = np.array([p.vector for p in self.points]).min(axis=0)
        self.box_center = (down_boundary + up_boundary) / 2
        self.box_radius = get_dist(self.box_center, down_boundary)

    # def check_intersection(self, center: np.ndarray, radius: float) -> bool:
    #     r1 = np.array([self.box_radius] * self.box_center.shape[0])
    #     r2 = np.array([radius] * center.shape[0])
    #     x = np.array([self.box_center - r1, center - r2]).max(axis=0)
    #     y = np.array([self.box_center + r1, center + r2]).min(axis=0)
    #     return (x <= y).all()

    def check_intersection(self, center: np.ndarray, radius: float) -> bool:
        return get_dist(self.box_center, center) <= self.box_radius + radius


class KDTree:

    def build_tree(
            self,
            points: typing.List[Point],
            axis: int,
            dimension: int,
            leaf_size: int,
            depth: int = 0
    ) -> typing.Optional[Node]:
        if len(points) == 0:
            return None
        pivot = float(np.median(np.array([p.vector[axis] for p in points])))
        points_number = len(points)

        if points_number <= leaf_size:
            return Node(axis, pivot, points_number, points=points)

        points.sort(key=lambda p: p.vector[axis])
        left_part = points[:points_number // 2]
        right_part = points[points_number // 2:]

        left_son = self.build_tree(left_part, (axis + 1) % dimension, dimension, leaf_size, depth + 1)
        right_son = self.build_tree(right_part, (axis + 1) % dimension, dimension, leaf_size, depth + 1)

        return Node(axis, pivot, points_number, left_son, right_son, points)

    def __init__(self, points: np.ndarray, leaf_size: int = 40):
        points = [Point(vector, i) for i, vector in enumerate(points)]
        points_number = len(points)
        assert points_number > 0
        dimension = points[0].vector.shape[0]
        assert dimension > 0
        self.root = self.build_tree(points, 0, dimension, leaf_size)
        self.cnt = 0

    def find_circle_boundary(self, point: np.ndarray, node: Node, k: int):
        node_to = node.left_son if point[node.axis] < node.axis_value else node.right_son
        if node_to and node_to.points_number >= k:
            return self.find_circle_boundary(point, node_to, k)
        node.points.sort(key=lambda p: p.get_dist(point))  # TODO: can be done in O(n)
        return node.points[min(k, len(node.points)) - 1].get_dist(point), node.points[:min(k, len(node.points))]

    def update_knn(
            self,
            node: Node,
            current_knn: typing.List[Point],
            k: int,
            center: np.ndarray,
            radius: float
    ) -> typing.Tuple[float, typing.List[Point]]:
        if not node or not node.check_intersection(center, radius):
            return radius, current_knn
        if not node.left_son and not node.right_son:
            current_knn += node.points
            current_knn.sort(key=lambda p: p.get_dist(center))
            if k < len(current_knn):
                current_knn = current_knn[:k]
            radius = current_knn[-1].get_dist(center)
            return radius, current_knn
        radius, current_knn = self.update_knn(node.left_son, current_knn, k, center, radius)
        radius, current_knn = self.update_knn(node.right_son, current_knn, k, center, radius)
        return radius, current_knn

    def find_knn(self, point: np.ndarray, k: int) -> typing.List[Point]:
        self.cnt += 1
        print(self.cnt)
        assert self.root.points_number >= k
        radius, knn = self.find_circle_boundary(point, self.root, k)
        radius, knn = self.update_knn(self.root, [], k, point, radius)
        return knn

    def query(self, xs: np.ndarray, k=1, return_distance=True):
        if self.root.points_number >= k:
            result = [self.find_knn(x, k) for x in xs]
        else:
            result = [self.root.points for _ in xs]
        if return_distance:
            return [[(p.id, p.get_dist(xs[i])) for p in knn] for i, knn in enumerate(result)]
        else:
            return [[p.id for p in knn] for knn in result]
