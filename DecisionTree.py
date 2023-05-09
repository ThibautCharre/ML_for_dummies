import numpy as np

# Decision tree is a prediction algorithm aiming at creating a model with a reduction of entropy.
# The main goal consists in creating nodes vertically, each node being able to separate features values improving the order of the Y datas.
# When all datas are categorized or the tree becomes too high, the algortithm has completely learned from the training dataset and is ready for prediction.


# Main steps of a tree
# 1 : Initialize a tree with a basic node
# 2 : Selection of a number (hyperparameter) of randomed features used for dataset split
# 3 : Looking for the best threshold (features unique values) and best feature susceptible split the original dataset and maximizing the information gain after having created 2 nodes
# 4 : Repeating this and stopping the process when the depth of the tree is > max_depth (hyperparameter) or number of samples in the dataset is > min_sample_spli (hyperparameter) 

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, max_depth=100, min_sample_split=2, nb_features=None):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.nb_features = nb_features
        self.root = None

    def fit(self, X, y):
        self.nb_features = X.shape[1] if not self.nb_features else min(X.shape[1], self.nb_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stop criterion
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split:
            leaf_name = self._most_common_value(y)
            return Node(value=leaf_name)

        # Find best splits
        feat_idxs = np.random.choice(range(X.shape[1]), self.nb_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_node = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right_node = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)
    
    def _most_common_value(self, y):
        ref, sum_ref = y[0], 0
        for label in np.unique(y):
            if np.sum(y == label) > sum_ref:
                ref, sum_ref = label, np.sum(y == label)
        return ref

    def _best_split(self, X, y, feat_idxs):
        base_gain = -1
        best_feat, best_threshold = None, None
        for feat_idx in feat_idxs:
            x_column = X[:, feat_idx]
            thresholds = np.unique(x_column)
            for thr in thresholds:
                new_gain = self._information_gain(x_column, y, thr)
                if new_gain > base_gain:
                    best_feat, best_threshold = feat_idx, thr
        return best_feat, best_threshold

    def _information_gain(self, x_column, y, threshold):

        # Calculate parent entropy
        parent_entropy = self._entropy(y)

        # Create Children
        left_idxs, right_idxs = self._split(x_column, threshold)
        if left_idxs == 0 or right_idxs == 0:
            return 0

        # Calculate weighted average entropy for children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l * e_l + n_r * e_r) / n

        # Calculate whole IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self, x_column, threshold):
        left_idxs = np.argwhere(x_column <= threshold).flatten()
        right_idxs = np.argwhere(x_column > threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, Node):
        if Node.is_leaf_node():
            return Node.value

        if x[Node.feature] <= Node.threshold:
            return self._traverse_tree(x, Node.left)
        else:
            return self._traverse_tree(x, Node.right)
