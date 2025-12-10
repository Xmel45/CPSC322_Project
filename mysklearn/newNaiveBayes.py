import numpy as np
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.num_mean = None
        self.num_var = None
        self.cat_prob = None
        self.priors = None
        self.num_idx = None
        self.cat_idx = None

    def fit(self, X, y, numeric_idx=None, categorical_idx=None):
        """
        X : 2D NumPy array of features
        y : 1D array of labels
        numeric_idx : list of column indices that are numeric
        categorical_idx : list of column indices that are categorical (one-hot or integer categories)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.num_idx = numeric_idx if numeric_idx else []
        self.cat_idx = categorical_idx if categorical_idx else []
        # Prepare storage
        self.num_mean = np.zeros((n_classes, len(self.num_idx)))
        self.num_var = np.zeros((n_classes, len(self.num_idx)))
        self.cat_prob = [{} for _ in range(n_classes)]
        self.priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0] / n_samples
            # Numeric features 
            if self.num_idx:
                self.num_mean[i, :] = X_c[:, self.num_idx].mean(axis=0)
                self.num_var[i, :] = X_c[:, self.num_idx].var(axis=0) + 1e-9  # prevent div by 0

            # Categorical features 
            for j in self.cat_idx:
                vals, counts = np.unique(X_c[:, j], return_counts=True)
                total = X_c.shape[0]
                # Store probability for each category in this feature
                self.cat_prob[i][j] = {v: (count / total) for v, count in zip(vals, counts)}

    def _gaussian_prob(self, x, mean, var):
        # Gaussian probability calculator
        return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean) ** 2 / (2 * var))

    def _predict_single(self, x):
        posteriors = []

        for i, c in enumerate(self.classes):
            posterior = np.log(self.priors[i])
            # Numeric features
            if self.num_idx:
                probs = self._gaussian_prob(x[self.num_idx], self.num_mean[i], self.num_var[i])
                posterior += np.sum(np.log(probs))
            # Categorical features
            for j in self.cat_idx:
                val = x[j]
                prob_dict = self.cat_prob[i][j]
                # If unseen category, assign small probability
                prob = prob_dict.get(val, 1e-6)
                posterior += np.log(prob)

            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

class OneHotEncoder:
    def __init__(self):
        self.categories_ = []
        self.mapping_ = {}
    def fit(self, data):
        self.categories_ = sorted(set(data))  # sort to keep order consistent
        self.mapping_ = {cat: i for i, cat in enumerate(self.categories_)}
        return self

    def transform(self, data):
        one_hot = np.zeros((len(data), len(self.categories_)), dtype=int)
        for idx, value in enumerate(data):
            if value not in self.mapping_:
                raise ValueError(f"Unknown category: {value}")
            one_hot[idx, self.mapping_[value]] = 1
        return one_hot

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


