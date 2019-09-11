from KDTree import KDTree
from plots import *


class KNearest:

    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        self.k = n_neighbors
        self.leaf_size = leaf_size
        self.kd_tree = None
        self.labels = None
        self.ys = None

    def fit(self, xs: list, ys: list):
        self.kd_tree = KDTree(np.array(xs), self.leaf_size)
        self.ys = ys
        self.labels = list(set(ys))

    def predict_probabilities(self, xs: list) -> np.ndarray:
        predicted_ys = self.kd_tree.query(xs, self.k, False)
        ps = [[]] * len(xs)
        for i in range(len(xs)):
            ps[i] = [0.0] * len(self.labels)
            for y in predicted_ys[i]:
                ps[i][self.ys[y]] += 1
            ps[i] = [p / len(predicted_ys[i]) for p in ps[i]]
        return np.array(ps)

    def predict(self, xs: list):
        return np.argmax(self.predict_probabilities(xs), axis=1).tolist()
