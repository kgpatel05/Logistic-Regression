import numpy as np
import sys

def sigmoid(z):
    """
    Compute the sigmoid function
    
    Args:
        z (numpy.ndarray): Input values
        
    Returns:
        numpy.ndarray: Sigmoid of input values
    """
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate, num_epochs):
    """
    Train a logistic regression model using gradient descent
    
    Args:
        X (numpy.ndarray): Training features of shape (n_samples, n_features)
        y (numpy.ndarray): Target values of shape (n_samples,)
        learning_rate (float): The step size for gradient descent
        num_epochs (int): Number of epochs for gradient descent
        
    Returns:
        tuple: (weights, bias) - Trained model parameters
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(num_epochs):
        # Forward pass
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)
        
        # Compute gradients
        dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1/n_samples) * np.sum(y_predicted - y)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
            
    return weights, bias

def load_data(filename):
    """
    Load data from file
    
    Args:
        filename (str): Path to the data file
        
    Returns:
        tuple: Features array and labels array
    """
    data = np.loadtxt(filename, delimiter=' ')
    X = data[:, :-1]  # All columns except last
    y = data[:, -1]   # Last column
    return X, y

def main():
    # Get command line arguments
    train_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    num_epochs = int(sys.argv[3])
    
    # Load training data and train model
    X, y = load_data(train_file)
    weights, bias = train_logistic_regression(X, y, learning_rate, num_epochs)
    
    # Print parameters (weights followed by bias) on a single line
    print(" ".join(map(str, np.concatenate([weights, [bias]]))))

if __name__ == "__main__":
    main()

