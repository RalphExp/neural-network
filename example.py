import numpy as np
from simple_nn import NeuralNetwork

# Example: XOR problem
if __name__ == "__main__":
    # Create training data for XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Initialize neural network with 2 input nodes, 4 hidden nodes, and 1 output node
    nn = NeuralNetwork(2, 4, 1)
    
    # Train the neural network
    print("Training the neural network...")
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    # Test the neural network
    print("\nTesting the neural network:")
    for i in range(len(X)):
        prediction = nn.predict(X[i].reshape(1, -1))
        print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {prediction[0][0]:.4f}")