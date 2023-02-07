import numpy as np
import random

# Class for Decision Tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        # Stop splitting when the maximum depth is reached
        if self.max_depth and depth >= self.max_depth:
            return Node(y)
        
        # Stop splitting when there is only one class left
        if len(set(y)) == 1:
            return Node(y)
        
        n_samples, n_features = X.shape
        n_classes = len(set(y))
        n_thresholds = n_samples * n_features * n_classes
        
        # Select a random feature and threshold to split the data
        feature = random.randint(0, n_features-1)
        thresholds = [random.uniform(min(X[:, feature]), max(X[:, feature])) for _ in range(n_thresholds)]
        
        best_gini = float('inf')
        best_split = None
        
        for threshold in thresholds:
            left_mask = X[:, feature] < threshold
            right_mask = X[:, feature] >= threshold
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            gini = self._gini(left_y, right_y, n_samples)
            
            if gini < best_gini:
                best_gini = gini
                best_split = (feature, threshold, left_mask, right_mask)
        
        if best_split is None:
            return Node(y)
        
        feature, threshold, left_mask, right_mask = best_split
        left_X = X[left_mask]
        right_X = X[right_mask]
        left_y = y[left_mask]
        right_y = y[right_mask]
        
        # Recursively build the tree
        left_child = self._build_tree(left_X, left_y, depth+
