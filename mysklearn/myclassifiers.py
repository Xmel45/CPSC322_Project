from mysklearn import myutils
"""
Programmer: Xavier Melancon
Class: CPSC 322-01 Fall 2025
Programming Assignment #6
Description: This program is a helper program for PA6 copied over from previous PAs with the addition of the Naive Bayes Classifier
"""


from mysklearn import myutils
import numpy as np

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
        self.X_train = X_train
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
        
        distances =[]
        neighbor_indices = []
        
        for iter in range(len(X_test)):
        
            instance_distances = []
            instance_neighbor_indices = []
            for x in range(len(self.X_train)):
                neighbor = X_test[iter]
                neighbor_distance = 0
                for j in range(len(neighbor)):
                    if(type(self.X_train[x][j])!=str):
                        neighbor_distance += ((self.X_train[x][j] - neighbor[j])**2)
                neighbor_distance = np.sqrt(neighbor_distance)
                instance_distances.append(neighbor_distance)


            sorted_indices = sorted(range(len(instance_distances)), key=lambda i: instance_distances[i])
            instance_neighbor_indices = sorted_indices[:self.n_neighbors]
            distances.append(instance_distances)
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
