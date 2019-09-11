from src.plots import *


# X, y = read_cancer_dataset('resources/cancer.csv')
# X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
# plot_precision_recall(X_train, y_train, X_test, y_test)
# plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10)


X, y = read_spam_dataset("resources/spam.csv")
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
# plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)
