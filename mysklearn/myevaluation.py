from mysklearn import myutils

import numpy as np # use numpy's random number generation

from mysklearn import myutils
"""
Programmer: Xavier Melancon
Class: CPSC 322-01 Fall 2025
Programming Assignment #6
Description: This program is a helper program for PA6 copied over from previous PAs with the addition of binary predictors for F1, Precision, and Recall
"""
def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    # Setup
    if random_state:
        np.random.seed(random_state)
    used_indeces = {}
    if shuffle: # This passes the unit test
        for i in range(len(X)):
                new_idx = np.random.randint(0,len(X))
                while (new_idx in used_indeces.values()):
                    new_idx = np.random.randint(0,len(X))
                used_indeces[i] = new_idx

    X_train, X_test, y_train, y_test = [],[],[],[]

    # Train Size
    if test_size < 1:
        train_size = int(len(X) * (1 - test_size))
    else:
        train_size = len(X) - test_size
    
    # Train Sets
    for i in range(train_size):

        if shuffle:
            X_train.append(X[used_indeces[i]])
            y_train.append(y[used_indeces[i]])

        else:
            X_train.append(X[i])
            y_train.append(y[i])

    # Test Sets
    for i in range(train_size,train_size+int(len(X)-train_size)):

        if shuffle:
            X_test.append(X[used_indeces[i]])
            y_test.append(y[used_indeces[i]])

        else:
            X_test.append(X[i])
            y_test.append(y[i])
        
    return X_train,X_test,y_train,y_test 


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):

    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    # Setup
    if random_state:
        np.random.seed(random_state)

    indices = [i for i in range(len(X))]

    if shuffle:
        np.random.shuffle(indices)
    
    folds, training_sets, test_sets = [],[],[]
    # Indices per fold and Overflow
    indices_per_fold = (len(X) // n_splits)
    overflow_samples = len(X) % n_splits

    # Train/Test Sets
    for i in range (n_splits):

        current_training_indices, current_test_indices = [],[]
        training_sets.append(current_training_indices)
        test_sets.append(current_test_indices)
        count = 0

        for j in indices:

            if overflow_samples>0:

                if count<indices_per_fold+1:

                    if j not in [item for sublist in test_sets for item in sublist]:

                        test_sets[i].append(j)
                        count += 1
                    else:
                        training_sets[i].append(j)

                else:
                        training_sets[i].append(j)
                
            else:
                if count<indices_per_fold:

                    if j not in [item for sublist in test_sets for item in sublist]:
                        test_sets[i].append(j)
                        count += 1
                    else:
                        training_sets[i].append(j)

                else:
                        training_sets[i].append(j)


        overflow_samples -= 1

    for i in range(n_splits):
        folds.append((training_sets[i],test_sets[i]))

    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Setup
    if random_state is not None:
        np.random.seed(random_state)

    indices = [i for i in range(len(X))]

    if shuffle:
        np.random.shuffle(indices)

    classes = sorted(set(y))

    # Group indices by class label (without defaultdict)
    class_indices = []
    for c in classes:
        class_indices.append([i for i, label in zip(indices, y) if label == c])

    
    # Folds
    folds = [[] for i in range(n_splits)]

    # Build Folds
    for strat_indices in class_indices:

        if shuffle:
            np.random.shuffle(strat_indices)
        # Indices per Fold
        n_samples = len(strat_indices)
        fold_sizes = [n_samples // n_splits + (1 if i < n_samples % n_splits else 0) for i in range(n_splits)]
        start = 0

        # Build Folds
        for i in range(len((fold_sizes))):
            end = start + fold_sizes[i]
            folds[i].extend(strat_indices[start:end])
            start = end

    # Final Folds
    final_folds = []
    all_indices = set(indices)

    for i in range(n_splits):

        test_idx = sorted(folds[i])
        train_idx = sorted(list(all_indices - set(test_idx)))
        final_folds.append((train_idx, test_idx))

    return final_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    # Setup
    if random_state:
        np.random.seed(random_state)

    X_sample, X_out_of_bag, y_sample, y_out_of_bag=[],[],[],[]

    if n_samples == None:
        n_samples = len(X)

    random_indices = [np.random.randint(0,len(X)) for i in range(n_samples)]

    # Build Sample Sets
    for i in range(n_samples):

        X_sample.append(X[random_indices[i]])

        if y is not None:
            y_sample.append(y[random_indices[i]])

    # Build Out of Bag Sets
    for i in range(len(X)):

        if i not in random_indices:
            X_out_of_bag.append(X[i])

            if y is not None:
                y_out_of_bag.append(y[i])
        
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag 


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Setup Empty Matrix
    matrix = []
    for i in range(len(labels)):
        matrix_row = [0 for i in range(len(labels))]
        matrix.append(matrix_row)
    
    # Fill Matrix
    for i in range(len(y_true)):
        true_idx = labels.index(y_true[i])
        pred_idx = labels.index(y_pred[i])
        matrix[true_idx][pred_idx] +=1
        
    return matrix 


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    # Setup
    score = 0

    # Accuracy
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            score += 1

    # Normalize        
    if normalize:
        score /= len(y_true)
    return score 

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    labels, pos_label = myutils.labels_poslabel(labels,pos_label,y_true)
    tp = fp = 0
    for i, val in enumerate(y_pred):
        if val == pos_label:
            if y_true[i] == pos_label:
                tp+=1
            else: fp+=1
    if (tp+fp)==0:
        return 0
    else:
        return tp/(tp+fp) # TODO: fix this

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    labels, pos_label = myutils.labels_poslabel(labels,pos_label,y_true)
    tp = fp = 0
    for i, val in enumerate(y_pred):
        if val == pos_label == y_true[i]:
            tp+=1
        elif pos_label == y_true[i] != val:
            fp+=1

    if (tp+fp)==0:
        return 0
    else:
        return tp/(fp+tp)


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    p = binary_precision_score(y_true,y_pred,labels,pos_label)
    r = binary_recall_score(y_true,y_pred,labels,pos_label)
    if (p+r)==0:
        return 0
    else:
        return 2*(p*r)/(p+r)
