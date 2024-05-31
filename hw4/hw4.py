import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values.

    Returns:
    - The Pearson correlation coefficient between the two columns.
    """
    r = 0.0
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_minus_mean_x = x - mean_x
    y_minus_mean_y = y - mean_y

    numerator = np.sum(x_minus_mean_x * y_minus_mean_y)
    denominator = np.sqrt(np.sum(x_minus_mean_x ** 2) * np.sum(y_minus_mean_y ** 2))

    r = numerator / denominator if denominator != 0 else 0
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).
    """
    # Drop non-numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    best_features = []
    pearson_correlations = [pearson_correlation(X_numeric.iloc[:, i], y) for i in range(X_numeric.shape[1])]
    # Sort indices by absolute correlation values and select top n_features indices
    best_feature_indices = np.argsort(np.array(pearson_correlations))[::-1][:n_features]
    best_features = X_numeric.columns[best_feature_indices]  # Get the feature names
    return best_features.tolist()

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        # set random seed
        np.random.seed(self.random_state)

        # Applies the bias trick to the input data
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize theta randomly
        self.theta = np.random.random(X.shape[1])
        self.thetas.append(self.theta)  # initialize theta

        # Perform LG
        for _ in range(self.n_iter):
            # Compute predictions and error
            h = self.sigmoid(X.dot(self.theta))
            errors = h - y

            # Update theta based on gradient
            gradient = X.T.dot(errors)
            self.theta -= self.eta * gradient

            # Save and check cost improvement for early stopping
            self.Js.append(self.compute_cost_LG(X, y))
            self.thetas.append(self.theta)

            if len(self.Js) >= 2 and abs(self.Js[-2] - self.Js[-1]) < self.eps:
                break

    def sigmoid(self, x):
        """
        Compute the sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))

    def compute_cost_LG(self, X, y):
        """
        Compute the cost function for Logistic Regression.
        """
        m = X.shape[0]
        h = self.sigmoid(X.dot(self.theta))

        one_label = - y.T.dot(np.log(h))
        zero_label = - (1 - y).T.dot(np.log(1 - h))

        # Compute cost
        J = (one_label + zero_label) / m
        return J


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        # Applies the bias trick to the input data
        X = np.c_[np.ones((X.shape[0], 1)), X]

        h = self.sigmoid(X.dot(self.theta))
        preds = np.where(h >= 0.5, 1, 0)
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None
    # set random seed
    np.random.seed(random_state)

    # shuffle the data
    combined = np.hstack((X, y.reshape(-1, 1)))  # Combine X and y
    np.random.shuffle(combined)  # Shuffle
    X_shuffled = combined[:, :-1]  # Separate the features
    y_shuffled = combined[:, -1]  # Separate the labels

    # Creates folds
    fold_size = len(X) // folds
    accuracies = []

    for i in range(folds):
        start_train = i * fold_size
        end_train = start_train + fold_size
        X_train = np.concatenate((X_shuffled[:start_train], X_shuffled[end_train:]), axis=0)
        y_train = np.concatenate((y_shuffled[:start_train], y_shuffled[end_train:]), axis=0)
        X_val = X_shuffled[start_train:end_train]
        y_val = y_shuffled[start_train:end_train]

        # train the model on this fold
        algo.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = algo.predict(X_val)
        accuracy = np.sum(y_val == y_pred) / len(y_val)
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """
    # Calculate the normal distribution pdf using the standard formula
    p = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * (np.exp(-0.5 * ((data - mu) / sigma) ** 2))
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        # initialize weights to be all equal
        self.weights = np.ones(self.k) / self.k

        # initialize sigmas to be the standard deviation of the data
        self.sigmas = np.full(self.k, np.std(data))

        # selecting random data points for initialize mus
        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indexes]

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        # Calculate the numerator of the responsibilities
        numerators = np.array([weight * norm_pdf(data, mu, sigma)
                               for (weight, mu, sigma) in zip(self.weights, self.mus, self.sigmas)])
        # Sum of the numerators for normalization
        sum_numerators = np.sum(numerators, axis=0 ,keepdims=True)
        self.responsibilities = numerators / sum_numerators

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        N = len(data)

        # Update the parameters for each Gaussian component
        for k in range(self.k):
            self.weights[k] = sum(self.responsibilities[k][i]
                                  for i in range(len(data))) / N

            self.mus[k] = sum(data[i] * self.responsibilities[k][i]
                              for i in range(len(data))) / (self.weights[k] * N)

            self.sigmas[k] = np.sqrt(sum(self.responsibilities[k][i] * np.square((data[i] - self.mus[k]))
                                         for i in range(len(data))) / (self.weights[k] * N))

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.costs.append(self.compute_minus_log_likelihood(data))

        # Perform EM
        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            self.costs.append(self.compute_minus_log_likelihood(data))
            if len(self.costs) >= 2 and np.abs(self.costs[-2] - self.costs[-1]) < self.eps:
                break

    def compute_minus_log_likelihood(self, data):
        weighted_pdfs = np.array([weight * norm_pdf(data, mu, sigma)
                               for (weight, mu, sigma) in zip(self.weights, self.mus, self.sigmas)])

        # Sum across all Gaussians for each data point
        total_pdf = np.sum(weighted_pdfs, axis=0)

        # Calculate the negative log-likelihood
        minus_log_likelihood = -np.sum(np.log(total_pdf))
        return minus_log_likelihood

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """
    weighted_pdfs = np.array([weight * norm_pdf(data, mu, sigma)
                              for (weight, mu, sigma) in zip(weights, mus, sigmas)])
    # Sum across all Gaussians for each data point
    pdf = np.sum(weighted_pdfs, axis=0)
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = []

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.classes = np.unique(y)
        self.EM_parameters = [[] for _ in self.classes]  # Initialize list of lists

        for i, cls in enumerate(self.classes):
            # Filter data points belonging to the current class
            X_given_cls = X[cls == y]

            # Compute and store prior probability for the current class
            self.prior.append(len(X_given_cls) / len(X))

            for feature_col in X_given_cls.T:
                # Fit a Gaussian Mixture Model to the feature column
                feature_em = EM(k=self.k, random_state=self.random_state)
                feature_em.fit(feature_col)

                # Retrieve and store the GMM parameters (weights, means, sigmas)
                weights, mus, sigmas = feature_em.get_dist_params()
                self.EM_parameters[i].append((weights, mus, sigmas))


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        for x in X:
            posteriors = []
            for cls_i, cls in enumerate(self.classes):
                likelihood_given_cls = self.compute_likelihood_given_cls(x, cls_i)
                posteriors.append(likelihood_given_cls * self.prior[cls_i])
            # Predict the class with the highest posterior probability
            preds.append(self.classes[np.argmax(posteriors)])
        return preds

    def compute_likelihood_given_cls(self, x, cls_i):
        likelihood_given_cls = 1  # Initialize the likelihood for the current class
        for feature_j_index, feature_j in enumerate(x):
            weight, mu, sigma = self.EM_parameters[cls_i][feature_j_index]
            # Compute the GMM PDF for the feature and multiply with the current likelihood
            likelihood_given_cls *= gmm_pdf(feature_j, weight, mu, sigma)
        return likelihood_given_cls


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    '''
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Fit Logistic Regression model with the best params you found earlier.
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(x_train, y_train)
    lor_model_train_preds = lor_model.predict(x_train)
    lor_model_test_preds = lor_model.predict(x_test)

    # Fit Naive Bayes model. Remember that you need to select the number of gaussians in the EM.
    bayes_model = NaiveBayesGaussian(k=k)
    bayes_model.fit(x_train, y_train)
    bayes_model_train_preds = bayes_model.predict(x_train)
    bayes_model_test_preds = bayes_model.predict(x_test)

    # Training and test accuracies for each model.
    lor_train_acc = np.sum(lor_model_train_preds == y_train) / len(y_train)
    lor_test_acc = np.sum(lor_model_test_preds == y_test) / len(y_test)
    bayes_train_acc = np.sum(bayes_model_train_preds == y_train) / len(y_train)
    bayes_test_acc = np.sum(bayes_model_test_preds == y_test) / len(y_test)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    # Parameters for dataset A (Naive Bayes works better)
    mean_a1 = [1, 1, 1]
    cov_a1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mean_a2 = [5, 5, 5]
    cov_a2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    data_a1 = multivariate_normal(mean=mean_a1, cov=cov_a1).rvs(size=100)
    data_a2 = multivariate_normal(mean=mean_a2, cov=cov_a2).rvs(size=100)

    dataset_a_features = np.vstack((data_a1, data_a2))
    dataset_a_labels = np.hstack((np.zeros(100), np.ones(100)))

    # Parameters for dataset B (Logistic Regression works better)
    mean_b1 = [2, 2, 2]
    cov_b1 = [[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]]
    mean_b2 = [4, 4, 4]
    cov_b2 = [[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]]

    data_b1 = multivariate_normal(mean=mean_b1, cov=cov_b1).rvs(size=100)
    data_b2 = multivariate_normal(mean=mean_b2, cov=cov_b2).rvs(size=100)

    dataset_b_features = np.vstack((data_b1, data_b2))
    dataset_b_labels = np.hstack((np.zeros(100), np.ones(100)))
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }