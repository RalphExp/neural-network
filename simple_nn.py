import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        # Backpropagation
        self.error = y - output
        self.delta2 = self.error * self.sigmoid_derivative(output)
        
        self.error_hidden = self.delta2.dot(self.W2.T)
        self.delta1 = self.error_hidden * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.W2 += self.a1.T.dot(self.delta2) * learning_rate
        self.b2 += np.sum(self.delta2, axis=0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(self.delta1) * learning_rate
        self.b1 += np.sum(self.delta1, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        # Training the neural network
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        # Make predictions
        return self.forward(X)