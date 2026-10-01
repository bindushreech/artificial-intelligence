import numpy as np


# -----------------------------
# Part A: Perceptron for AND
# -----------------------------

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        x = np.insert(x, 0, 1)
        summation = np.dot(self.weights, x)
        return 1 if summation >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction

                xi = np.insert(xi, 0, 1)
                self.weights += self.lr * error * xi


# AND gate
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(X_and, y_and)

print("AND Gate Predictions:")

for x in X_and:
    print(x, "->", perceptron.predict(x))


# ------------------------------------
# Part B: Backpropagation for XOR
# ------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def train_backpropagation(X, y, epochs=5000, lr=0.1):

    input_size = X.shape[1]
    hidden_size = 3
    output_size = 1

    np.random.seed(42)

    weights_input_hidden = np.random.uniform(
        -1, 1, (input_size, hidden_size)
    )

    weights_hidden_output = np.random.uniform(
        -1, 1, (hidden_size, output_size)
    )

    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

    for _ in range(epochs):

        # Forward propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(
            hidden_output,
            weights_hidden_output
        ) + bias_output

        final_output = sigmoid(final_input)

        # Error
        error = y - final_output

        # Backpropagation
        d_output = error * sigmoid_derivative(final_output)

        d_hidden = (
            d_output.dot(weights_hidden_output.T)
            * sigmoid_derivative(hidden_output)
        )

        # Update weights
        weights_hidden_output += (
            hidden_output.T.dot(d_output) * lr
        )

        weights_input_hidden += (
            X.T.dot(d_hidden) * lr
        )

        bias_output += np.sum(
            d_output,
            axis=0,
            keepdims=True
        ) * lr

        bias_hidden += np.sum(
            d_hidden,
            axis=0,
            keepdims=True
        ) * lr

    return (
        weights_input_hidden,
        weights_hidden_output,
        bias_hidden,
        bias_output
    )


# XOR gate
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
])


weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = \
    train_backpropagation(X_xor, y_xor)


# Final prediction
hidden_output = sigmoid(
    np.dot(X_xor, weights_input_hidden) + bias_hidden
)

final_output = sigmoid(
    np.dot(hidden_output, weights_hidden_output) + bias_output
)

print("\nXOR Gate Predictions:")

for x, prediction in zip(X_xor, final_output):
    print(x, "->", round(float(prediction[0]), 4))