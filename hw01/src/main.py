from src.plots import *


def test_cancer():
    xs, y = read_cancer_dataset('resources/cancer.csv')
    x_train, y_train, x_test, y_test = train_test_split(xs, y, 0.9)
    plot_precision_recall(x_train, y_train, x_test, y_test)
    plot_roc_curve(x_train, y_train, x_test, y_test, max_k=10)


def test_spam():
    xs, y = read_spam_dataset("resources/spam.csv")
    x_train, y_train, x_test, y_test = train_test_split(xs, y, 0.9)
    plot_precision_recall(x_train, y_train, x_test, y_test, max_k=20)
    plot_roc_curve(x_train, y_train, x_test, y_test, max_k=20)


if __name__ == '__main__':
    test_cancer()
    test_spam()
