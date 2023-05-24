from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class RandomForest:

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, nb_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.nb_features = nb_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_sample_split=self.min_samples_split,
                                nb_features=self.nb_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_value(pred) for pred in tree_preds])
        return predictions

    def _most_common_value(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value


# # Check if the model is good with the breast cancer dataset
# dataset = load_breast_cancer()
# X, y = dataset.data, dataset.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#
# # Model prediction
# clf = RandomForest()
# clf.fit(X_train, y_train)
# pred_clf = clf.predict(X_test)
#
#
# # Accuracy score
# def accuracy(y_pred, y_test):
#     return np.sum(y_pred == y_test) / len(y_test)
#
#
# print(f'Accuracy score is : {accuracy(pred_clf, y_test)}')
