import typing

from sklearn.neighbors import KDTree

from src.utils import *


class KMeans:

    def __init__(self, n_clusters: int, init: str = 'random', max_iter: int = 300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

        self.centroid_crds = None
        self.centroid_labels = None

        self.kd_tree = None

    def sample_init(self, xs: np.ndarray, ys: np.ndarray):
        centroid_ids = np.arange(xs.shape[0])
        np.random.shuffle(centroid_ids)
        self.centroid_crds = np.array([xs[i] for i in centroid_ids])[:self.n_clusters]
        self.centroid_labels = np.array([ys[i] for i in centroid_ids])[:self.n_clusters]

    def random_init(self, shape: typing.Tuple):
        self.centroid_crds = np.random.random(shape)[:self.n_clusters]
        self.centroid_labels = np.arange(self.n_clusters)

    def k_means_plus_plus_init(self, xs: np.ndarray, ys: np.ndarray):
        centroid_ids = [np.random.randint(0, xs.shape[0])]
        for i in range(1, self.n_clusters):
            kd_tree = KDTree(np.array([xs[i] for i in centroid_ids]))
            dists, _ = kd_tree.query(X=xs)
            dists = np.array([d ** 2 for d in dists[:, 0]])

            next_d = np.random.uniform(0, dists.sum())
            next_c = 0
            cur_sum_d = 0
            while next_c < xs.shape[0] and cur_sum_d + dists[next_c] < next_d:
                cur_sum_d += dists[next_c]
                next_c += 1

            centroid_ids.append(next_c)

        self.centroid_crds = np.array([xs[i] for i in centroid_ids])
        self.centroid_labels = np.array([ys[i] for i in centroid_ids])

    def shift_centroid(self):
        for _ in range(self.max_iter):
            

    def fit(self, xs: np.ndarray, ys: np.ndarray = None):
        if ys is None:
            ys = np.arange(xs.shape[0])

        if self.init == 'random':
            self.random_init(xs.shape)
        elif self.init == 'sample':
            self.sample_init(xs, ys)
        elif self.init == 'k-means++':
            self.k_means_plus_plus_init(xs, ys)
        else:
            raise Exception(f'Unrecognised init type: {self.init}')

    def predict(self, xs: np.ndarray):
        while True:
            if not self.kd_tree:
                self.kd_tree = KDTree(self.centroid_crds)
            _, cs = self.kd_tree.query(xs)
            result = cs[:, 0]
            if np.unique(result).shape[0] != self.n_clusters:
                continue
            else:
                return np.array([self.centroid_labels[c] for c in result])
