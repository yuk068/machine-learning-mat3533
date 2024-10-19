import numpy as np

def step_function(z):
    """Step function to convert input to binary output (0 or 1)."""
    return np.where(z >= 0, 1, 0)

def forward(weights, x):
    """Compute the linear combination of inputs and weights."""
    x_with_bias = np.c_[np.ones(x.shape[0]), x]  # Add bias term
    z = np.dot(x_with_bias, weights)
    return step_function(z)  # Apply step function for binary output

def perceptron_loss(y_true, y_pred):
    """Perceptron loss function based on misclassification."""
    return np.sum(y_true != y_pred)

def update_weights(weights, x, y, learning_rate):
    """Update the weights using the perceptron update rule."""
    # Forward pass
    y_pred = forward(weights, x)

    # Compute the error
    error = y - y_pred

    # Update weights if there is a misclassification
    gradient = np.dot(np.c_[np.ones(x.shape[0]), x].T, error)  # Gradient based on error
    weights += learning_rate * gradient  # Update weights based on the error
    return weights

def train(weights, x, y, learning_rate, epochs):
    """Train the perceptron model."""
    for epoch in range(epochs):
        for i in range(len(y)):
            weights = update_weights(weights, x[i:i+1], y[i:i+1], learning_rate)
        # Optional: Print perceptron loss every 10 epochs
        # if epoch % 10 == 0:
        #     y_pred = forward(weights, x)
        #     loss = perceptron_loss(y, y_pred)
    return weights

def initialize_weights(input_dim):
    """Initialize weights (including bias)."""
    np.random.seed(42)
    return np.random.randn(input_dim + 1)  # +1 for the bias term

def perceptron_predict(X_train, y_train, X_test, max_iter=1000, learning_rate=0.01):
    """Train the perceptron and predict on test data."""
    # Initialize weights
    weights = initialize_weights(X_train.shape[1])

    # Train the model
    weights = train(weights, X_train, y_train, learning_rate, max_iter)

    # Make predictions on test set
    y_pred = forward(weights, X_test)
    return y_pred