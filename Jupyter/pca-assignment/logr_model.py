import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = np.clip(z, -500, 500)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(weights, x, classification_type='binary'):
    # Add bias term (column of 1s) to the input features
    x_with_bias = np.c_[np.ones(x.shape[0]), x]

    # Compute the linear combination of inputs and weights
    z = np.dot(x_with_bias, weights)

    if classification_type == 'binary':
        return sigmoid(z)
    elif classification_type == 'softmax':
        return softmax(z)

def log_loss(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_loss(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Convert y_true to one-hot encoding if it is not
    y_one_hot = np.eye(y_pred.shape[1])[y_true]
    return -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))

def update_weights(weights, x, y, learning_rate, regularization=None, reg_lambda=0.01, classification_type='binary'):
    # Forward pass
    y_pred = forward(weights, x, classification_type)

    # Compute the error (gradient of loss w.r.t. predictions)
    if classification_type == 'binary':
        error = y_pred - y
    elif classification_type == 'softmax':
        y_one_hot = np.eye(y_pred.shape[1])[y]
        error = y_pred - y_one_hot

    # Compute gradient (with respect to weights)
    gradient = np.dot(np.c_[np.ones(x.shape[0]), x].T, error) / x.shape[0]

    # Apply L1 or L2 regularization to the gradient (excluding bias term)
    if regularization == 'l2':
        gradient[1:] += reg_lambda * weights[1:] / x.shape[0]
    elif regularization == 'l1':
        gradient[1:] += reg_lambda * np.sign(weights[1:]) / x.shape[0]

    # Update weights
    weights -= learning_rate * gradient
    return weights

def initialize_weights(input_dim, num_classes=1, seed=42):
    np.random.seed(seed)
    return np.random.normal(0, 1, (input_dim + 1, num_classes))

def check_weight_convergence(old_weights, new_weights, tolerance=1e-4):
    return np.all(np.abs(new_weights - old_weights) < tolerance)

def train(weights, x, y, learning_rate, epochs, regularization=None, reg_lambda=0.01, classification_type='binary'):
    old_weights = np.copy(weights)
    for epoch in range(epochs):
        for i in range(len(y)):
            weights = update_weights(weights, x[i:i+1], y[i:i+1], learning_rate, regularization, reg_lambda, classification_type)

        # Check for convergence every 20 iterations
        if (epoch + 1) % 20 == 0:
            if check_weight_convergence(old_weights, weights):
                # Uncomment the line below to know when the model converged
                # print(f"Converged after {epoch + 1} epochs")
                return weights
            old_weights = np.copy(weights)

        # Uncomment the following section to print the loss over epochs
        # y_pred = forward(weights, x, classification_type)
        # if classification_type == 'binary':
        #     loss = log_loss(y, y_pred)
        # elif classification_type == 'softmax':
        #     loss = cross_entropy_loss(y, y_pred)
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # Uncomment the line below to know if the model reached maximum iteration count
    # print(f"Reached maximum number of epochs: {epochs}")
    return weights

def logistic_regression_predict(X_train, y_train, X_test, max_iter=10000, learning_rate=0.05, type='binary', regularization=None, reg_lambda=0.01):
    # Weight initialization
    if type == 'binary':
        weights = initialize_weights(X_train.shape[1])
    elif type == 'softmax':
        num_classes = len(np.unique(y_train))
        weights = initialize_weights(X_train.shape[1], num_classes)

    # Train the model
    weights = train(weights, X_train, y_train, learning_rate, max_iter, regularization, reg_lambda, type)

    # Make predictions on test set
    y_pred_probs = forward(weights, X_test, type)

    # Convert probabilities to class predictions
    if type == 'binary':
        y_pred = (y_pred_probs >= 0.5).astype(int)
    elif type == 'softmax':
        y_pred = np.argmax(y_pred_probs, axis=1)

    return np.array(y_pred)