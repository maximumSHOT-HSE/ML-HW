import numpy as np

from src.KDTree import KDTree


def check_correctness():

    def true_closest(xx_train, xx_test, k):
        result = []
        for x0 in xx_test:
            bests = list(sorted([(i, np.linalg.norm(x - x0)) for i, x in enumerate(xx_train)], key=lambda x: x[1]))
            bests = [i for i, d in bests]
            result.append(bests[:min(k, len(bests))])
        return result

    x_train = np.random.randn(100, 20)
    x_test = np.random.randn(100, 20)
    tree = KDTree(x_train, leaf_size=2)
    predicted = tree.query(x_test, k=10, return_distance=False)
    true = true_closest(x_train, x_test, k=10)

    predicted = [list(sorted(l)) for l in predicted]
    true = [list(sorted(l)) for l in true]

    print(np.array(true))
    print('===================')
    print(np.array(predicted))

    if np.sum(np.abs(np.array(np.array(predicted).shape) - np.array(np.array(true).shape))) != 0:
        print("Wrong shape")
    else:
        errors = sum([1 for row1, row2 in zip(predicted, true) for i1, i2 in zip(row1, row2) if i1 != i2])
        if errors > 0:
            print(errors, "errors")
        else:
            print('OK')


def execute_max_test():
    x_train = np.random.randn(1000, 20)
    x_test = np.random.randn(1000, 20)
    tree = KDTree(x_train, leaf_size=2)
    predicted = tree.query(x_test, k=4)

    predicted = [list(sorted(l)) for l in predicted]

    print(np.array(predicted))


if __name__ == '__main__':
    check_correctness()
    execute_max_test()
