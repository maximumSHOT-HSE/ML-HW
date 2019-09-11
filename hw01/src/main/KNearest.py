import sys

import matplotlib.pyplot as plt
import numpy as np

from src.main.KDTree import KDTree
from src.main.utils import *


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


def plot_precision_recall(x_train: list, y_train: list, x_test: list, y_test: list, max_k=30):
    ks = list(range(1, max_k + 1))
    classes = len(np.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k in ks:
        classifier = KNearest(k)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in range(classes):
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(ys) - 0.01, np.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    plot(ks, recalls, "Recall")
    plot(ks, precisions, "Precision")
    plot(ks, [accuracies], "Accuracy", legend=False)


def plot_roc_curve(x_train, y_train, x_test, y_test, max_k=30):
    positive_samples = sum(1 for y in y_test if y == 0)
    ks = list(range(1, max_k + 1))
    curves_tpr = []
    curves_fpr = []
    colors = []
    for k in ks:
        colors.append([k / ks[-1], 0, 1 - k / ks[-1]])
        knearest = KNearest(k)
        knearest.fit(x_train, y_train)
        p_pred = [p[0] for p in knearest.predict_probabilities(x_test)]
        tpr = []
        fpr = []
        for w in np.arange(-0.01, 1.02, 0.01):
            y_pred = [(0 if p > w else 1) for p in p_pred]
            tpr.append(
                sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples
            )
            fpr.append(
                sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples)
            )
        curves_tpr.append(tpr)
        curves_fpr.append(fpr)
    plt.figure(figsize=(7, 7))
    for tpr, fpr, c in zip(curves_tpr, curves_fpr, colors):
        plt.plot(fpr, tpr, color=c)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


# X, y = read_cancer_dataset('../resources/cancer.csv')
# X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
# plot_precision_recall(X_train, y_train, X_test, y_test)
# plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10)

sys.setrecursionlimit(2000)

X, y = read_spam_dataset("../resources/spam.csv")
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.5)
plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
# plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)
