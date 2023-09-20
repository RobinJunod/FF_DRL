import numpy as np

# Define the neural network architecture with one hidden layer
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
num_epochs = 10000

# Initialize weights and biases
np.random.seed(0)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Activation function (sigmoid) for the last layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# ReLu
def ReLu(x):
    return x if x > 0 else 0
        
# Forward pass
def forward(X):
    # Input to hidden layer
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    # Hidden to output layer
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    return hidden_input, hidden_output, output_input, predicted_output

# Backpropagation
def backward(X, Y, hidden_input, hidden_output, output_input, predicted_output):
    # Compute the loss
    loss = 0.5 * np.mean((Y - predicted_output) ** 2)

    # Compute gradients
    output_error = (predicted_output - Y) * predicted_output * (1 - predicted_output)
    hidden_error = np.dot(output_error, weights_hidden_output.T) * hidden_output * (1 - hidden_output)

    d_weights_hidden_output = np.dot(hidden_output.T, output_error)
    d_bias_output = np.sum(output_error, axis=0, keepdims=True)
    d_weights_input_hidden = np.dot(X.T, hidden_error)
    d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

    # Update weights and biases
    weights_hidden_output -= learning_rate * d_weights_hidden_output
    bias_output -= learning_rate * d_bias_output
    weights_input_hidden -= learning_rate * d_weights_input_hidden
    bias_hidden -= learning_rate * d_bias_hidden

    return loss

# Generate sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Training loop
for epoch in range(num_epochs):
    hidden_input, hidden_output, output_input, predicted_output = forward(X)
    loss = backward(X, Y, hidden_input, hidden_output, output_input, predicted_output)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Testing
_, _, _, predictions = forward(X)
print("Predictions:")
print(predictions)
