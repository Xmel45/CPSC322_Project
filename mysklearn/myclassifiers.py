from mysklearn import myutils
"""
Programmer: Xavier Melancon
Class: CPSC 322-01 Fall 2025
Programming Assignment #6
Description: This program is a helper program for PA6 copied over from previous PAs with the addition of the Naive Bayes Classifier
"""


from mysklearn import myutils
import numpy as np
import random
from collections import Counter
from mysklearn import myevaluation

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = np.array(X_train)
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        X_test = np.array(X_test)
        distances = []
        neighbor_indices = []
        
        # 1. Outer Loop: Iterate through each test instance
        for test_instance in X_test:
            
            # --- Vectorized Distance Calculation ---
            # 2. Subtract the single test instance (test_instance) from ALL training instances (self.X_train)
            #    Shape: (N_train_samples, n_features)
            diff = self.X_train - test_instance 
            
            # 3. Square the differences (element-wise)
            #    Shape: (N_train_samples, n_features)
            squared_diff = diff ** 2
            
            # 4. Sum the squared differences across the features (axis=1) to get the squared Euclidean distance
            #    This results in a 1D array of distances: (N_train_samples,)
            sum_of_squares = np.sum(squared_diff, axis=1)
            
            # 5. Take the square root to get the final Euclidean distance
            instance_distances = np.sqrt(sum_of_squares)

            sorted_indices = np.argsort(instance_distances)
            instance_neighbor_indices = sorted_indices[:self.n_neighbors].tolist()

            k_distances = instance_distances[instance_neighbor_indices].tolist()
            distances.append(k_distances)
            neighbor_indices.append(instance_neighbor_indices)
            
        return distances, neighbor_indices 

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        distances,neighbor_indices = self.kneighbors(X_test)
        for x in range(len(X_test)):
            classes = dict()
            for i in range(len(neighbor_indices[0])):
                if self.y_train[neighbor_indices[x][i]] in classes:
                    classes[self.y_train[neighbor_indices[x][i]]] +=1
                else:
                    classes[self.y_train[neighbor_indices[x][i]]] = 1
            y_predicted.append(max(classes,key=classes.get))
            
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        labels = {}
        for y in y_train:
            if y in labels:
                labels[y]+=1
            else:
                labels[y]=1
        self.most_common_label =(max(labels,key=labels.get))
        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in range(len(X_test)):
            y_predicted.append(self.most_common_label)
        return y_predicted # TODO: fix this


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None
        self.attributes = None
        self.classes = None
    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        self.classes = sorted(set(y_train))
        encoded_classes = {c:i for i,c in enumerate(self.classes)}
        y_encoded = [encoded_classes[i] for i in y_train]

        priors = [y_encoded.count(cls)/len(y_encoded) for cls in range(len(self.classes))]

        
        num_attr = len(X_train[0])
        self.attributes = [ sorted(set(i)) for i in zip(*X_train)]
        encoded_attrs = [{val:i for i,val in enumerate(vals)} for vals in self.attributes]
        x_encoded = [[encoded_attrs[i][val] for i,val in enumerate(row)] for row in X_train]
        counts = [[[0.0 for i in range(len(self.attributes[i]))] for i in range(num_attr)] for j in range(len(self.classes))]

        for x,y in zip(x_encoded,y_encoded):
            for i,val in enumerate(x):
                counts[y][i][val] +=1


        conditionals = [ [ [ (counts[cls][attr][idx])/sum(counts[cls][attr]) for idx in range(len(self.attributes[attr]))] for attr in range(num_attr)] for cls in range(len(self.classes))]

        
        self.priors = priors
        self.conditionals = conditionals

        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        

        y_predicted = []
        encoded_attrs = [{val:i for i,val in enumerate(vals)} for vals in self.attributes]
        x_encoded = [[encoded_attrs[i][val] for i,val in enumerate(row)] for row in X_test]
        y_predicted = []

        for x_new in x_encoded:

            posteriors = []
            for c in range(len(self.classes)):
                prob = self.priors[c]

                for j, idx in enumerate(x_new):
                    prob *= self.conditionals[c][j][idx]

                posteriors.append(prob)

            best_class_index = posteriors.index(max(posteriors))
            y_predicted.append(self.classes[best_class_index])

        return y_predicted
    

class Node:
    """Represents a node in the decision tree."""
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """Initialize a tree node.
        
        Args:
            feature_index (int): Index of the feature to split on (None for leaf nodes)
            threshold: The threshold value for the split (None for leaf nodes)
            left (Node): Left subtree
            right (Node): Right subtree
            value: Class label for leaf nodes
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class MyDecisionTreeClassifier:
    """A basic decision tree classifier using entropy and information gain."""
    
    def __init__(self, max_depth=None, max_features=None):
        """Initialize the decision tree.
        
        Args:
            max_depth (int): Maximum depth of the tree
            max_features (int): Number of features to randomly sample at each split
        """
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None
        self.n_features = None
        self.n_classes = None
    
    def fit(self, X_train, y_train):
        """Build the decision tree using training data.
        
        Args:
            X_train (list of lists): Training features
            y_train (list): Training labels
        """
        self.n_features = len(X_train[0]) if X_train else 0
        self.n_classes = len(set(y_train))
        self.root = self._build_tree(X_train, y_train, depth=0)
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree.
        
        Args:
            X (list of lists): Features
            y (list): Labels
            depth (int): Current depth in the tree
            
        Returns:
            Node: The root of the constructed subtree
        """
        # Base cases: stop if we have only one class, reached max depth, or no samples
        if len(set(y)) == 1:
            # All samples belong to one class
            return Node(value=y[0])
        
        if self.max_depth is not None and depth >= self.max_depth:
            # Reached max depth - create leaf with majority class
            return Node(value=self._get_majority_class(y))
        
        if len(X) == 0 or len(y) == 0:
            return Node(value=None)
        
        # Find the best split
        best_split = self._find_best_split(X, y)
        
        if best_split is None or best_split['info_gain'] <= 0:
            # No good split found - create leaf with majority class
            return Node(value=self._get_majority_class(y))
        
        # Recursively build left and right subtrees
        left_X, left_y = best_split['left_X'], best_split['left_y']
        right_X, right_y = best_split['right_X'], best_split['right_y']
        
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)
        
        # Create decision node
        return Node(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree
        )
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on.
        
        Args:
            X (list of lists): Features
            y (list): Labels
            
        Returns:
            dict: Dictionary containing split information or None if no split found
        """
        best_split = None
        best_info_gain = 0
        
        # Determine which features to consider
        if self.max_features is None:
            features_to_try = list(range(self.n_features))
        else:
            # Randomly select max_features features
            features_to_try = random.sample(range(self.n_features), 
                                           min(self.max_features, self.n_features))
        
        # Try splitting on each selected feature
        for feature_idx in features_to_try:
            # Get all unique values for this feature
            feature_values = [X[i][feature_idx] for i in range(len(X))]
            unique_values = sorted(set(feature_values))
            
            # Try each unique value as a potential threshold
            for threshold in unique_values:
                # Split the data
                left_X, left_y, right_X, right_y = self._split_data(X, y, feature_idx, threshold)
                
                # Skip if split doesn't separate the data
                if len(left_X) == 0 or len(right_X) == 0:
                    continue
                
                # Calculate information gain
                info_gain = self._calculate_information_gain(y, left_y, right_y)
                
                # Update best split if this is better
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        'feature_index': feature_idx,
                        'threshold': threshold,
                        'left_X': left_X,
                        'left_y': left_y,
                        'right_X': right_X,
                        'right_y': right_y,
                        'info_gain': info_gain
                    }
        
        return best_split
    
    def _split_data(self, X, y, feature_idx, threshold):
        """Split the data based on a feature and threshold.
        
        Args:
            X (list of lists): Features
            y (list): Labels
            feature_idx (int): Index of feature to split on
            threshold: Threshold value
            
        Returns:
            tuple: (left_X, left_y, right_X, right_y)
        """
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i in range(len(X)):
            if X[i][feature_idx] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y
    
    def _calculate_information_gain(self, parent_y, left_y, right_y):
        """Calculate information gain from a split.
        
        Args:
            parent_y (list): Labels before split
            left_y (list): Labels in left child
            right_y (list): Labels in right child
            
        Returns:
            float: Information gain value
        """
        parent_entropy = self._calculate_entropy(parent_y)
        
        # Weighted average of child entropies
        n_parent = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        if n_parent == 0:
            return 0
        
        left_entropy = self._calculate_entropy(left_y)
        right_entropy = self._calculate_entropy(right_y)
        
        weighted_child_entropy = (n_left / n_parent) * left_entropy + \
                                (n_right / n_parent) * right_entropy
        
        info_gain = parent_entropy - weighted_child_entropy
        return info_gain
    
    def _calculate_entropy(self, y):
        """Calculate entropy for a set of labels using log2.
        
        Args:
            y (list): Labels
            
        Returns:
            float: Entropy value
        """
        if len(y) == 0:
            return 0
        
        import math
        
        # Count occurrences of each class
        class_counts = Counter(y)
        entropy = 0
        
        for count in class_counts.values():
            probability = count / len(y)
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _get_majority_class(self, y):
        """Get the most common class label.
        
        Args:
            y (list): Labels
            
        Returns:
            The most common class label
        """
        if len(y) == 0:
            return None
        
        class_counts = Counter(y)
        return class_counts.most_common(1)[0][0]
    
    def predict(self, X_test):
        """Make predictions for test instances.
        
        Args:
            X_test (list of lists): Test features
            
        Returns:
            list: Predicted class labels
        """
        predictions = []
        for sample in X_test:
            prediction = self._traverse_tree(sample, self.root)
            predictions.append(prediction)
        
        return predictions
    
    def _traverse_tree(self, sample, node):
        """Traverse the tree to make a prediction for a single sample.
        
        Args:
            sample (list): Feature values for a single sample
            node (Node): Current node in the tree
            
        Returns:
            The predicted class label
        """
        if node.value is not None:
            # Leaf node - return the class label
            return node.value
        
        # Decision node - check the feature value and traverse accordingly
        feature_value = sample[node.feature_index]
        
        if feature_value <= node.threshold:
            return self._traverse_tree(sample, node.left)
        else:
            return self._traverse_tree(sample, node.right)


class MyRandomForestClassifier:
    """A basic random forest classifier using bootstrapping and majority voting.
    
    This implementation follows the project specifications:
    - Generates N random decision trees using bootstrapping
    - At each node, randomly selects F features as candidates to partition on
    - Selects the M most accurate trees using validation sets
    - Uses majority voting for predictions
    """
    
    def __init__(self, n_trees=10, max_depth=None, max_features=None, m_trees=None):
        """Initialize the random forest classifier.
        
        Args:
            n_trees (int): Number of trees to generate (N)
            max_depth (int): Maximum depth for each tree
            max_features (int): Number of random features to try at each split (F)
            m_trees (int): Number of most accurate trees to keep (M). 
                          If None, keeps all trees.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.m_trees = m_trees if m_trees is not None else n_trees
        self.trees = []
        self.classes = None
    
    def fit(self, X_train, y_train):
        """Fit the random forest classifier.
        
        Generate N random decision trees using bootstrapping over the training set.
        Select the M most accurate trees using their validation accuracy (out-of-bag samples).
        
        Args:
            X_train (list of lists): Training features
            y_train (list): Training labels
        """
        self.classes = list(set(y_train))
        
        tree_scores = []
        
        # Generate N trees using bootstrapping
        for i in range(self.n_trees):
            # Use myevaluation.bootstrap_sample to create bootstrap training and OOB validation sets
            X_bootstrap, X_oob, y_bootstrap, y_oob = myevaluation.bootstrap_sample(
                X_train, y_train, random_state=None
            )
            
            # Train tree on bootstrap sample
            tree = MyDecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Calculate validation accuracy on OOB samples
            if len(X_oob) > 0:
                oob_predictions = tree.predict(X_oob)
                accuracy = myevaluation.accuracy_score(y_oob, oob_predictions)
            else:
                # If no OOB samples, use bootstrap accuracy
                bootstrap_predictions = tree.predict(X_bootstrap)
                accuracy = myevaluation.accuracy_score(y_bootstrap, bootstrap_predictions)
            
            tree_scores.append((tree, accuracy))
        
        # Sort trees by accuracy and keep the M best trees
        tree_scores.sort(key=lambda x: x[1], reverse=True)
        self.trees = [tree for tree, _ in tree_scores[:self.m_trees]]
    
    def predict(self, X_test):
        """Make predictions using majority voting across all trees.
        
        Args:
            X_test (list of lists): Test features
            
        Returns:
            list: Predicted class labels
        """
        if len(self.trees) == 0:
            raise ValueError("Forest has no trees. Call fit() first.")
        
        all_predictions = []
        
        # Get predictions from each tree
        for tree in self.trees:
            tree_predictions = tree.predict(X_test)
            all_predictions.append(tree_predictions)
        
        # Majority voting - for each test sample, find the most common prediction
        final_predictions = []
        for sample_idx in range(len(X_test)):
            # Collect all predictions for this sample
            sample_votes = [all_predictions[tree_idx][sample_idx] 
                           for tree_idx in range(len(self.trees))]
            
            # Find the most common vote
            vote_counts = Counter(sample_votes)
            majority_vote = vote_counts.most_common(1)[0][0]
            final_predictions.append(majority_vote)
        
        return final_predictions
