import numpy as np
import matplotlib.pyplot as plt

class FeedForwardNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_layer = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
        return self.output_layer

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.hidden_error = np.dot(self.output_delta, self.weights2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_layer)
        self.weights2 += np.dot(self.hidden_layer.T, self.output_delta)
        self.bias2 += np.sum(self.output_delta, axis=0, keepdims=True)
        self.weights1 += np.dot(X.T, self.hidden_delta)
        self.bias1 += np.sum(self.hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

        print("Training complete!")

    def predict(self, X):
        return self.forward(X)

# Generate random synthetic data for binary classification
def generate_data(num_samples):
    X = np.random.randn(num_samples, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int).reshape(-1, 1)
    return X, y

# Generate training and testing data
train_X, train_y = generate_data(100)
test_X, test_y = generate_data(20)


# Create and train the neural network
input_size = 2
hidden_size = 4
output_size = 1
epochs = 1000
learning_rate = 0.1

nn = FeedForwardNN(input_size, hidden_size, output_size)
nn.train(train_X, train_y, epochs, learning_rate)

# Evaluate the neural network on test data
predictions = nn.predict(test_X)
accuracy = np.mean((predictions > 0.5) == test_y)
print(f"Test accuracy: {accuracy:.2f}")


# Visualize the decision boundary
x_min, x_max = train_X[:, 0].min() - 0.5, train_X[:, 0].max() + 0.5
y_min, y_max = train_X[:, 1].min() - 0.5, train_X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.Spectral)
plt.title("Decision Boundary")
plt.show()
