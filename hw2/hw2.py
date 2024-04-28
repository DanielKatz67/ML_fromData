import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = \
        {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    labels = data[:, -1]
    label_counts, counts = np.unique(labels, return_counts=True)
    total_labels = counts.sum()

    # Calculate Gini impurity
    gini = 1.0 - np.sum((counts / total_labels) ** 2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    total_labels = counts.sum()
    probs = counts / total_labels

    # Calculate entropy, adding small epsilon to avoid log2(0)
    entropy = - np.sum(probs * np.log2(probs + np.finfo(float).eps))
    return entropy

class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        if self.data is not None and self.data.size > 0:
            labels = self.data[:, -1]
            label_counts, counts = np.unique(labels, return_counts=True)
            max_index = np.argmax(counts)  # Get the index of the maximum count
            pred = label_counts[max_index]  # Get the label with the maximum count
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        if self.feature != -1 and n_total_sample > 0:
            proportion_of_samples = self.data.shape[0] / n_total_sample
            goodness = self.goodness_of_split(self.feature)
            self.feature_importance = proportion_of_samples * goodness
        else:
            self.feature_importance = 0
        return self.feature_importance

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset

        # Extract the column of data corresponding to the feature index
        feature_column = self.data[:, feature]
        # Get unique values of this feature
        unique_values = np.unique(feature_column)
        groups = {value: self.data[feature_column == value] for value in unique_values}

        # Compute weighted impurity of each group
        n_total_sample = self.data.shape[0]
        groups_impurity = sum(data_subset.shape[0] / n_total_sample * self.impurity_func(data_subset) for data_subset in groups.values())

        root_impurity = self.impurity_func(self.data)
        information_gain = root_impurity - groups_impurity

        if self.gain_ratio:  # calculate gainRatio
            if self.impurity_func.__name__ != 'calc_entropy':
                raise ValueError("Gain ratio requires the impurity function to be entropy.")
            splitInformation = self.__split_information(feature)
            if splitInformation == 0:
                return 0, groups  # Avoid division by zero
            goodness = information_gain / splitInformation
        else :  # calculate the regular Goodness of Split
            goodness = information_gain

        return goodness, groups

    def __split_information(self, feature) :
        """
        Helper function to calculate Gain Ratio for Attribute with many values
        """
        feature_column = self.data[:, feature]
        _, counts = np.unique(feature_column, return_counts=True)
        total = counts.sum()
        probs = counts / total
        # Adding a epsilon to probabilities to avoid log2(0)
        return - np.sum(probs * np.log2(probs + np.finfo(float).eps))

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Stop if maximum depth is reached
        if self.depth == self.max_depth:
            self.terminal = True  # the node is a leaf
            return

        best_goodness = -1
        best_feature = None
        best_groups = None

        # Evaluate all possible features to find the best one for splitting
        for feature_index in range(self.data.shape[1] - 1):  # Exclude the label column
            goodness, groups = self.goodness_of_split(feature_index)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature_index
                best_groups_data = groups

        # Set the best feature found
        self.feature = best_feature

        # Apply the best split if any good split found
        # best_goodness=0 means that father_goodness = children_goodness -> no need to split
        if best_feature is not None and best_goodness > 0 and self.__chi_square_Test(best_groups_data):
            for value, data_subset in best_groups_data.items():
                if data_subset.size > 0:
                    child = DecisionNode(data_subset, self.impurity_func, depth=self.depth+1, chi=self.chi,
                                         max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                    self.add_child(child, value)
        else:
            self.terminal = True  # the node is a leaf

    def __chi_square_Test(self, feature_values_data):
        """
        check if the split is not random according to the best feature values.
        """
        # If chi is 1, skip the chi-square test and accept the split
        if self.chi == 1:
            return True

        # Calculate the degrees of freedom and the chi-square value
        degree_of_freedom = (len(feature_values_data) - 1) * (len(np.unique(self.data[:, -1])) - 1)
        value_from_chi_table = chi_table[degree_of_freedom][self.chi]

        # Calculate the chi-square statistic
        chi_square = 0
        size = len(self.data[:, -1])
        labels_count = {label: np.sum(self.data[:, -1] == label) for label in np.unique(self.data[:, -1])}

        for feature_value, sub_data in feature_values_data.items():
            sub_len = len(sub_data)
            # Count occurrences of each label in each subset
            inner_label_count = {label: np.sum(sub_data[:, -1] == label) for label in np.unique(sub_data[:, -1])}

            for label, count in labels_count.items():
                sampled = inner_label_count.get(label, 0)  # Count occurrences of each label in each subset
                expectation = sub_len * (count / size)  # Expected count if the split was random

                chi_square += ((sampled - expectation) ** 2) / expectation

        return chi_square > value_from_chi_table



class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio
        self.root = None  # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        queue_of_nodes = []
        self.root = DecisionNode(self.data, self.impurity_func, depth=0, chi=self.chi,
                                 max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        queue_of_nodes.append(self.root)

        while queue_of_nodes:
            current_node = queue_of_nodes.pop(0)
            current_node.feature_importance = current_node.calc_feature_importance(len(self.root.data))

            # Check if the current node is pure (the depth and the goodness are checked in split function)
            if self.__is_pure(current_node.data):
                current_node.terminal = True
                continue

            # Perform splitting on the current node if it's not pure
            current_node.split()

            # Add children nodes to the list
            for child in current_node.children:
                queue_of_nodes.append(child)

    def __is_pure(self, data):
        """
        Check if all data belong to the same class.
        """
        labels = data[:, -1]
        return len(np.unique(labels)) == 1

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        current_node = self.root
        while not current_node.terminal:
            if current_node.children == [] or current_node.feature == -1 :
                break

            instance_feature_value = instance[current_node.feature]
            # Find the child node that corresponds to the feature value
            found_child = False
            for child, value in zip(current_node.children, current_node.children_values):
                if value == instance_feature_value:
                    current_node = child
                    found_child = True
                    break
            if not found_child:
                break  # If no matching child is found, stop at the last matched node

        return current_node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        right_pred = 0
        for instance in dataset:
            # Predict the label for each instance using the predict method of the tree
            prediction = self.predict(instance)
            if prediction == instance[-1]:
                right_pred += 1
        # Calculate the accuracy as the percentage of correctly predicted instances
        accuracy = (right_pred / len(dataset)) * 100
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None

    # Iterate over a range of depths to explore their effect on model accuracy
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        root = DecisionTree(X_train, calc_entropy, max_depth=max_depth, gain_ratio=True)
        root.build_tree()
        training.append(root.calc_accuracy(X_train))
        validation.append(root.calc_accuracy(X_validation))

    return training, validation

def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    # Iterate over predefined chi values to see the effect of chi-square pruning
    for p_value in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        root = DecisionTree(X_train, calc_entropy, chi=p_value, gain_ratio=True)
        root.build_tree()
        chi_training_acc.append(root.calc_accuracy(X_train))
        chi_validation_acc.append(root.calc_accuracy(X_test))
        depth.append(__get_tree_depth(root.root))

    return chi_training_acc, chi_validation_acc, depth

def __get_tree_depth(node):
    """
    Recursively calculate the maximum depth of the tree starting from the given node.

    Args:
        node (DecisionNode): The node from which to calculate the depth.

    Returns:
        int: The maximum depth from the given node to the deepest leaf node.
    """
    if node.terminal:
        return node.depth
    else:
        return max(__get_tree_depth(child) for child in node.children)

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    queue_of_nodes = []
    queue_of_nodes.append(node)
    n_nodes = 0

    while queue_of_nodes:
        current_node = queue_of_nodes.pop(0)  # Pop the first node in the queue
        n_nodes += 1
        if not current_node.terminal:
            # Add children nodes to the list
            for child in current_node.children:
                queue_of_nodes.append(child)

    return n_nodes







