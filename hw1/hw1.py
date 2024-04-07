
# imports
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X_mean = np.mean(X ,axis=0)
    X_max = np.max(X ,axis=0)
    X_min = np.min(X ,axis=0)
    y_mean = np.mean(y)
    y_max = np.max(y)
    y_min = np.min(y)

    # Perform mean normalization
    X_norm = (X - X_mean) / (X_max - X_min)
    y_norm = (y - y_mean) / (y_max - y_min)

    return X_norm, y_norm
def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    # Add column of ones to X
    return np.c_[np.ones((X.shape[0], 1)), X]

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    J = 0  # We use J for the cost.
    m = X.shape[0]

    # Compute the hypothesis - predicted prices
    h = X.dot(theta)
    squared_loss = (h - y) ** 2
    # Compute cost
    J = squared_loss.sum() / (2 * m)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    m = X.shape[0]
    temp_theta = theta.copy()

    # Perform gradient descent
    for i in range(num_iters):
        # Compute predictions and error
        h = X.dot(theta)
        errors = h - y

        # Update theta based on gradient
        gradient = errors.dot(X)
        temp_theta = temp_theta - alpha * gradient / m
        theta = temp_theta

        # Save cost in history for convergence check
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    # Calculate X^T * X
    XTX = X.T.dot(X)
    # Inverse of X^T * X
    XTX_inv = np.linalg.inv(XTX)
    pinv_X = XTX_inv.dot(X.T)
    # Calculate optimal theta using the pseudoinverse of X
    pinv_theta = pinv_X.dot(y)

    return pinv_theta
def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = X.shape[0]
    temp_theta = theta.copy()

    # Perform gradient descent with early stopping
    for i in range(num_iters):
        # Compute predictions and error
        h = X.dot(theta)
        errors = h - y

        # Update theta based on gradient
        gradient = errors.dot(X)
        temp_theta = temp_theta - alpha * gradient / m
        theta = temp_theta

        # Save and check cost improvement for early stopping
        J_history.append(compute_cost(X, y, theta))
        if len(J_history) >= 2 and J_history[-2]-J_history[-1] < 1e-8:
            break

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    np.random.seed(42)
    random_theta = np.random.random(size=X_train.shape[1])

    # Find the best alpha
    for alpha in alphas:
        # Train the model and get the validation cost
        min_theta, ignore = efficient_gradient_descent(
            X_train, y_train, random_theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, min_theta)

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices

    Implement forward feature selection using the following guidelines:
        1. Start with an empty set of selected features.
        2. For each feature not yet in the selected set, do the following:
            1. Add the feature to the selected set temporarily.
            2. Train a model using the current set of selected features and evaluate
               its performance by calculating the cost or error on a validation set.
            3. Remove the temporarily added feature from the selected set.
        3. Choose the feature that resulted in the best model performance and permanently add it to the selected set.
        4. Repeat steps 2-3 until you have 5 features (not including the bias).
    """
    selected_features = []
    n = X_train.shape[1]  # Total number of features

    while len(selected_features) < 5:
        min_cost = np.inf
        best_feature = None
        np.random.seed(42)
        random_theta = np.random.random(size=len(selected_features)+2) # 1 extra for tetha_0, 1 extra for the new feature

        # Evaluate each feature not yet selected
        for j in range(n):
            if j not in selected_features:
                # Temporary feature set including the new candidate feature
                temp_features = selected_features + [j]

                # Extract the data with selected features and apply the bias trick
                X_train_selected = apply_bias_trick(X_train[:, temp_features])
                X_val_selected = apply_bias_trick(X_val[:, temp_features])

                min_theta, J_history = efficient_gradient_descent(X_train_selected, y_train, random_theta, best_alpha, iterations)
                cost = compute_cost(X_val_selected, y_val, min_theta)
                # Update best feature based on minimum cost
                if cost < min_cost:
                    min_cost = cost
                    best_feature = j

        # Add the best feature found in this iteration to the final selected features list
        if best_feature is not None:
            selected_features.append(best_feature)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """
    df_poly = df.copy()
    feature_names = df.columns
    square_features = {}  # Dictionary to store new square features

    for i in range(len(feature_names)):
        for j in range(i, len(feature_names)):
            # Creating feature names for squares and interactions
            if i == j :
                feature_name = f'{feature_names[i]}^2'
            else :
                feature_name = f'{feature_names[i]}*{feature_names[j]}'
            # Compute and store the new feature
            square_features[feature_name] = df_poly[feature_names[i]] * df_poly[feature_names[j]]

    # Concatenate all new features into the original DataFrame
    return pd.concat([df_poly, pd.DataFrame(square_features, index=df_poly.index)], axis=1)


