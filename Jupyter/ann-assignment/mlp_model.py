import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, random_seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        if random_seed is not None:
            np.random.seed(random_seed)
        self.weights_input_hidden = self.init_weights(input_size, hidden_size)
        self.weights_hidden_output = self.init_weights(hidden_size, output_size)

    def init_weights(self, in_size, out_size):
        limit = np.sqrt(6 / (in_size + out_size))
        return np.random.uniform(-limit, limit, (in_size, out_size))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = self.relu(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output_layer_output = self.softmax(self.output_layer_input)
        return self.output_layer_output

    def cost(self, Y, output):
        return -np.mean(Y * np.log(output + 1e-9))

    def backprop(self, X, Y, output):
        output_error = output - Y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * (self.hidden_layer_output > 0)
        return output_error, hidden_error

    def update_weights(self, X, output_error, hidden_error):
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_layer_output.T, output_error)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_error)

    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            cost = self.cost(Y, output)
            output_error, hidden_error = self.backprop(X, Y, output)
            self.update_weights(X, output_error, hidden_error)
            if epoch % 100 == 0:  
                print(f'Epoch {epoch}, Cost: {cost:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
