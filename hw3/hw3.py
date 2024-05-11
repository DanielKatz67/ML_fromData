
import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.08,
            (0, 0, 1): 0.02,
            (0, 1, 0): 0.12,
            (0, 1, 1): 0.08,
            (1, 0, 0): 0.12,
            (1, 0, 1): 0.08,
            (1, 1, 0): 0.18,
            (1, 1, 1): 0.32,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for (x, y), p_xy in X_Y.items():
            # Check if the product of P(X=x) and P(Y=y) equals the joint probability P(X=x, Y=y)
            if not np.isclose(p_xy, X[x] * Y[y]):
                return True  # Dependent
        return False  # Independent



    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        # Iterate over all combinations of values for X, Y, and C
        for c in C:
            for x in X:
                for y in Y:
                    # P(X=x |C=c)
                    p_x_given_c = X_C[(x, c)] / C[c]
                    # P(Y=y|C=c)
                    p_y_given_c = Y_C[(y, c)] / C[c]
                    # P(X=x, Y=y|C=c)
                    p_xy_given_c = X_Y_C[(x, y, c)] / C[c]
                    # Check if the P(X=x, Y=y|C=c) = P(X=x |C=c) * P(X=x, Y=y|C=c)
                    if not np.isclose(p_xy_given_c, p_x_given_c * p_y_given_c):
                        return False  # They are conditionally dependent given C

        return True  # Conditionally independent given C
def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    # Calculate the logarithm of the Poisson probability mass function (pmf)
    p = (rate ** k) * (np.exp(-rate)) / np.math.factorial(k)
    log_p = np.log(p)
    return log_p
def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """

    likelihoods = np.zeros(len(rates))
    # Calculate log likelihood for each rate
    for i, rate in enumerate(rates):
        # Sum log PMFs across all samples for this rate
        likelihoods[i] = np.sum([poisson_log_pmf(k, rate) for k in samples])
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
    # The index of the maximum log-likelihood
    max_likelihood_index = np.argmax(likelihoods)
    # Return the rate corresponding to the maximum log-likelihood
    rate = rates[max_likelihood_index]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    # The MLE for a Poisson distribution's rate lambda is the sample mean
    mean = np.mean(samples)
    return mean
def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    # Calculate the normal distribution pdf using the standard formula
    p = (1 / np.sqrt(2 * np.pi * std**2)) * (np.exp(-0.5 * ((x-mean)/std)**2))
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.filtered_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.mean = np.mean(self.filtered_data, axis=0) if self.filtered_data.size > 0 else None
        self.std = np.std(self.filtered_data, axis=0) if self.filtered_data.size > 0 else None

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        # The prior is the proportion of the class instances in the total dataset
        prior = self.filtered_data.size / self.dataset.size
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        # Compute the product of the normal pdfs for each feature in x
        likelihood = np.prod([normal_pdf(x, mean, std) for mean, std in zip(self.mean, self.std)])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        # Calculate the posterior using Bayes' Theorem, ignoring the probability of x (denominator)
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Compare the two posterior probabilities and return the class with the higher probability
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0
        else:
            pred = 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    classified_correct = 0

    for instance in test_set:
        # Predict the class excluding the actual class label
        pred = map_classifier.predict(instance[: -1])
        # Check if the prediction matches the actual class label
        if pred == instance[-1]:
            classified_correct += 1

    acc = classified_correct / len(test_set) if len(test_set) > 0 else 0
    return acc
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = len(x)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    x_dif_m = x-mean
    exp = np.exp(-0.5 * np.dot(x_dif_m.T, np.dot(inv, x_dif_m)))

    # Calculate the full density function using the normalization factor and the exponent.
    pdf = (1 / np.sqrt((2 * np.pi) ** d * det)) * exp
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.filtered_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.mean = np.mean(self.filtered_data, axis=0) if self.filtered_data.size > 0 else None
        self.cov = np.cov(self.filtered_data.T)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        # The prior is the proportion of the class instances in the total dataset
        prior = self.filtered_data.size / self.dataset.size
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        # Use the multivariate normal PDF to calculate the likelihood
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        # Compute posterior by multiplying the prior and the likelihood
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Determine the class by comparing the priors of the two distributions
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            pred = 0
        else:
            pred = 1
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Determine the class by comparing the likelihoods calculated for the instance in both distributions
        if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x):
            pred = 0
        else:
            pred = 1
        return pred

EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.filtered_data = dataset[dataset[:, -1] == class_value][:, :-1]

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        # The prior is the proportion of the class instances in the total dataset
        prior = self.filtered_data.size / self.dataset.size
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1.0
        n_i = len(self.filtered_data)  # Total number of instances in the class

        for feature_index, feature_label in enumerate(x):
            # Extract unique labels and their counts for the current feature in the class data
            feature_labels, counts = np.unique(self.filtered_data[:, feature_index], return_counts=True)
            V_j = len(feature_labels)  # Number of possible unique values of the feature

            if feature_label in feature_labels:
                instance_label_index = np.where(feature_labels == feature_label)[0]
                n_i_j = counts[instance_label_index]
                # Apply Laplace smoothing
                likelihood *= (n_i_j + 1) / (n_i + V_j)
            else:
                # Use EPSILLON for unseen labels in the training set
                likelihood *= EPSILLON

        return likelihood

        # example (from the slides) :
        # feature_index = 0
        # instance_lable = male
        # feature_labels = [Male, Female]
        # counts = [3, 3]

        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        # Multiply the prior probability by the likelihood of the instance
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior

class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Return the class that has the higher posterior probability
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0
        else:
            pred = 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        classified_correct = 0

        for instance in test_set:
            # Predict the class of each instance in the test set
            pred = self.predict(instance[: -1])  # Ignore the actual class label
            if pred == instance[-1]:
                # Check if the prediction matches the actual class label
                classified_correct += 1

        # Calculate the accuracy as the proportion of correctly classified instances
        acc = classified_correct / len(test_set) if len(test_set) > 0 else 0
        return acc


