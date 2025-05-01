import numpy as np
import sys

def sigmoid(z):
    """
    Numerically stable sigmoid function implementation.
    Clips inputs to avoid overflow and uses a stable computation approach.
    """
    z = np.clip(z, -500, 500)  # prevent overflow
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def compute_accuracy(X, y, weights, bias):
    preds = sigmoid(np.dot(X, weights) + bias) >= 0.5
    return np.mean(preds == y)


def train_logistic_regression(X_train, y_train, learning_rate, epochs, X_dev=None, y_dev=None):
    n_samples, n_features = X_train.shape
    # Initialize parameters in [-0.01, 0.01] randomly
    weights = np.random.uniform(-0.01, 0.01, size=n_features)
    bias = np.random.uniform(-0.01, 0.01)

    train_hist = []
    dev_hist = [] if X_dev is not None else None

    for epoch in range(epochs):
        # Forward pass
        logits = np.dot(X_train, weights) + bias
        preds = sigmoid(logits)
        # Compute gradients
        errors = preds - y_train
        grad_w = np.dot(X_train.T, errors) / n_samples
        grad_b = np.mean(errors)
        # Update parameters
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        # Record accuracies
        train_hist.append(compute_accuracy(X_train, y_train, weights, bias))
        if X_dev is not None:
            dev_hist.append(compute_accuracy(X_dev, y_dev, weights, bias))

    return weights, bias, train_hist, dev_hist

# Command-line Interface
# Usage: python main.py TRAIN_FILE LEARNING_RATE NUM_EPOCHS
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("wrong format input")
        sys.exit(1)

    train_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    epochs = int(sys.argv[3])

    #can be removed after testing
    np.random.seed(0)

    # Load and prepare data
    data = np.loadtxt(train_file)
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # Train model and output parameters
    weights, bias, _, _ = train_logistic_regression(X, y, learning_rate, epochs)
    params = np.concatenate([weights, np.array([bias])])
    print(" ".join(map(str, params)))