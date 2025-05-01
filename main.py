import numpy as np
import sys

#-------------------------------------------------------
#--------------MAIN IMPLEMENTATION----------------------
#-------------------------------------------------------
def sigmoid(z):
    """
    Sigmoid function

    params:
        z (numpy.ndarray): Input values

    return:
        numpy.ndarray: Sigmoid of input values
    """
    # needed to add thsi so that we could avoid overflow errors
    z = np.clip(z, -500, 500)

    # Use the stable version of sigmoid
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def compute_accuracy(X, y, weights, bias):
    """
    Compute accuracy for current model parameters

    params:
        X (numpy.ndarray): Features
        y (numpy.ndarray): True labels
        weights (numpy.ndarray): Model weights
        bias (float): Model bias

    return:
        float: Accuracy score
    """
    predictions = sigmoid(np.dot(X, weights) + bias) >= 0.5
    return np.mean(predictions == y)


def train_logistic_regression(X, y, learning_rate, num_epochs):
    """
    Train logistic regression with simple random initialization.
    """
    n_samples, n_features = X.shape

    # initialize weights and bias randomly in [-0.01, 0.01]
    weights = np.random.uniform(-0.01, 0.01, size=n_features)
    bias = np.random.uniform(-0.01, 0.01)

    for epoch in range(num_epochs):
        # forward pass
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        # compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        # parameter update
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, None


def load_data(filename):
    """
    Load data

    params:
        filename (str): Path to the data file

    return:
        tuple: Features array and labels array
    """
    data = np.loadtxt(filename, delimiter=' ')
    X = data[:, :-1]  # All columns except last
    y = data[:, -1]  # Last column
    return X, y


def main():
    train_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    num_epochs = int(sys.argv[3])

    X, y = load_data(train_file)
    weights, bias, _ = train_logistic_regression(X, y, learning_rate, num_epochs)
    print(" ".join(map(str, np.concatenate([weights, [bias]]))))


if __name__ == "__main__":
    main()
