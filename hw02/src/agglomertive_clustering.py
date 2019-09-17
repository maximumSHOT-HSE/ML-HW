import numpy as np

from src.dsu import DisjointSetUnion


class AgglomerativeClustering:
    def __init__(self, n_clusters=16, linkage='average'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, xs, ys=None):
        n_xs = xs.shape[0]
        dsu = DisjointSetUnion(n_xs)
        dists = np.zeros((n_xs, n_xs))
        for i in range(n_xs):
            for j in range(i):
                dists[j][i] = dists[i][j] = np.linalg.norm(xs[i] - xs[j])
        for _ in range(self.n_clusters, n_xs):
            min_dist = 0
            i = -1
            j = -1
            for qi in range(n_xs):
                if dsu.find(qi) != qi:
                    continue
                for qj in range(qi):
                    if dsu.find(qj) != qj:
                        continue
                    if i == -1 or min_dist > dists[qi][qj]:
                        min_dist = dists[qi][qj]
                        i = qi
                        j = qj
            c_i = dsu.size[i]
            c_j = dsu.size[j]
            dsu.merge(i, j)
            k = dsu.find(i)
            for q in range(n_xs):
                if q == i or q == j or dsu.find(q) != q:
                    continue
                d_was_i = dists[q][i]
                d_was_j = dists[q][j]
                if self.linkage == 'average':
                    d_now_k = (d_was_i * c_i + d_was_j * c_j) / (c_i + c_j)
                elif self.linkage == 'single':
                    d_now_k = min(d_was_i, d_was_j)
                elif self.linkage == 'complete':
                    d_now_k = max(d_was_i, d_was_j)
                else:
                    raise Exception(f'Unrecognized linkage type: {self.linkage}')
                dists[q][k] = dists[k][q] = d_now_k

        if ys is None:
            current_cluster_label = 0
            ys = [0] * n_xs
            for i in range(n_xs):
                if dsu.find(i) == i:
                    ys[i] = current_cluster_label
                    current_cluster_label += 1

        return [ys[dsu.find(p)] for p in range(n_xs)]
