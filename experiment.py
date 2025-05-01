import numpy as np
import matplotlib.pyplot as plt
import time
from main import load_data, train_logistic_regression, compute_accuracy, sigmoid

'''
def plot_accuracy_vs_epoch(train_file, dev_file, learning_rates, num_epochs, num_runs=5):
    """
    Plot mean ± std (and later min/max) accuracy vs. epoch for train & dev sets.
    """
    X_train, y_train = load_data(train_file)
    X_dev, y_dev   = load_data(dev_file)

    plt.figure(figsize=(10,6))
    epochs = np.arange(1, num_epochs+1)

    for lr in learning_rates:
        train_runs = []
        dev_runs   = []
        for _ in range(num_runs):
            # random init
            w = np.random.uniform(-0.01, 0.01, size=X_train.shape[1])
            b = np.random.uniform(-0.01, 0.01)

            tr_acc = []
            dv_acc = []
            for _ in range(num_epochs):
                # one gradient‐descent step on train
                z = X_train.dot(w) + b
                ŷ = sigmoid(z)
                dw = (1/len(X_train)) * X_train.T.dot(ŷ - y_train)
                db = (1/len(X_train)) * (ŷ - y_train).sum()
                w -= lr * dw
                b -= lr * db

                # record both train & dev accuracy each epoch
                tr_acc.append(compute_accuracy(X_train, y_train, w, b))
                dv_acc.append(compute_accuracy(X_dev,   y_dev,   w, b))

            train_runs.append(tr_acc)
            dev_runs.append(dv_acc)

        # convert to arrays: shape (num_runs, num_epochs)
        train_runs = np.array(train_runs)
        dev_runs   = np.array(dev_runs)

        # compute statistics
        train_mean = train_runs.mean(axis=0)
        train_std  = train_runs.std(axis=0)
        dev_mean   = dev_runs.mean(axis=0)
        dev_std    = dev_runs.std(axis=0)

        # plot mean ± std
        plt.plot(epochs, train_mean, '--', label=f"train lr={lr}")
        plt.fill_between(epochs,
                         train_mean - train_std,
                         train_mean + train_std,
                         alpha=0.2)
        plt.plot(epochs, dev_mean, '-', label=f"dev   lr={lr}")
        plt.fill_between(epochs,
                         dev_mean - dev_std,
                         dev_mean + dev_std,
                         alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Dev Accuracy by Epoch")  # evaluating on both train & dev :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_epoch.png")
    plt.close()
'''

def plot_accuracy_vs_epoch(train_file, dev_file, learning_rates, num_epochs, num_runs=5):
    """
    Plot mean ± std and min→max accuracy vs. epoch for train & dev sets.
    """
    X_train, y_train = load_data(train_file)
    X_dev, y_dev     = load_data(dev_file)

    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)

    for lr in learning_rates:
        train_runs = []
        dev_runs   = []

        for _ in range(num_runs):
            # random init
            w = np.random.uniform(-0.01, 0.01, size=X_train.shape[1])
            b = np.random.uniform(-0.01, 0.01)

            tr_acc = []
            dv_acc = []
            for _ in range(num_epochs):
                # one gradient‐descent step on train
                z = X_train.dot(w) + b
                ŷ = sigmoid(z)
                dw = (1/len(X_train)) * X_train.T.dot(ŷ - y_train)
                db = (1/len(X_train)) * (ŷ - y_train).sum()
                w -= lr * dw
                b -= lr * db

                # record both train & dev accuracy each epoch
                tr_acc.append(compute_accuracy(X_train, y_train, w, b))
                dv_acc.append(compute_accuracy(X_dev,   y_dev,   w, b))

            train_runs.append(tr_acc)
            dev_runs.append(dv_acc)

        # convert to arrays: shape (num_runs, num_epochs)
        train_runs = np.array(train_runs)
        dev_runs   = np.array(dev_runs)

        # compute statistics
        train_mean = train_runs.mean(axis=0)
        train_std  = train_runs.std(axis=0)
        train_min  = train_runs.min(axis=0)
        train_max  = train_runs.max(axis=0)

        dev_mean   = dev_runs.mean(axis=0)
        dev_std    = dev_runs.std(axis=0)
        dev_min    = dev_runs.min(axis=0)
        dev_max    = dev_runs.max(axis=0)

        # plot train mean
        plt.plot(epochs, train_mean, '--', label=f"train lr={lr}")
        # shade full min→max range behind
        plt.fill_between(epochs, train_min, train_max, alpha=0.1)
        # shade ±1 std around mean
        plt.fill_between(epochs,
                         train_mean - train_std,
                         train_mean + train_std,
                         alpha=0.2)

        # plot dev mean
        plt.plot(epochs, dev_mean, '-', label=f"dev   lr={lr}")
        # shade full min→max range behind
        plt.fill_between(epochs, dev_min, dev_max, alpha=0.1)
        # shade ±1 std around mean
        plt.fill_between(epochs,
                         dev_mean - dev_std,
                         dev_mean + dev_std,
                         alpha=0.07)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Dev Accuracy by Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_epoch.png")
    plt.close()

'''
def train_sgd_logistic_regression(X, y, learning_rate, num_epochs, batch_size):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    accuracies = []

    start_time = time.time()
    total_flops = 0

    for epoch in range(num_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i + batch_size]
            batch_y = y_shuffled[i:i + batch_size]
            batch_size_actual = len(batch_X)

            linear_model = np.dot(batch_X, weights) + bias
            y_predicted = sigmoid(linear_model)

            dw = (1 / batch_size_actual) * np.dot(batch_X.T, (y_predicted - batch_y))
            db = (1 / batch_size_actual) * np.sum(y_predicted - batch_y)

            weights -= learning_rate * dw
            bias -= learning_rate * db

            total_flops += batch_size_actual * (4 * n_features + 2)

        accuracy = compute_accuracy(X, y, weights, bias)
        accuracies.append(accuracy)

    wall_time = time.time() - start_time
    metrics = {
        'wall_time': wall_time,
        'total_flops': total_flops,
        'flops_per_epoch': total_flops / num_epochs
    }

    return weights, bias, accuracies, metrics
'''

'''
def analyze_sgd_performance(train_file, learning_rate, num_epochs, batch_sizes):
    """
    Analyze SGD performance for different batch sizes

    params:
        train_file: Path to training data
        learning_rate: Learning rate
        num_epochs: Number of epochs
        batch_sizes: List of batch sizes to try
    """
    X, y = load_data(train_file)
    n_samples = len(X)

    all_metrics = []
    all_accuracies = []

    with open('sgd_analysis.txt', 'w') as f:
        f.write("SGD Performance Analysis\n")
        f.write("=" * 50 + "\n\n")

        for batch_size in batch_sizes:
            f.write(f"\nTraining with batch size: {batch_size}\n")
            f.write("-" * 30 + "\n")

            weights, bias, accuracies, metrics = train_sgd_logistic_regression(
                X, y, learning_rate, num_epochs, batch_size
            )

            all_metrics.append({
                'batch_size': batch_size,
                **metrics
            })
            all_accuracies.append(accuracies)

            f.write("\nEpoch-by-epoch accuracy:\n")
            f.write(f"{'Epoch':>6} | {'Accuracy':>12}\n")
            f.write("-" * 25 + "\n")
            for epoch, acc in enumerate(accuracies, 1):
                f.write(f"{epoch:>6} | {acc:>12.4f}\n")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for i, batch_size in enumerate(batch_sizes):
        plt.plot(range(1, num_epochs + 1), all_accuracies[i],
                 label=f'Batch Size={batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Convergence by Batch Size')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    wall_times = [m['wall_time'] for m in all_metrics]
    plt.semilogx(batch_sizes, wall_times, '-o')
    plt.xlabel('Batch Size (log scale)')
    plt.ylabel('Wall Time (seconds)')
    plt.title('Wall Time vs Batch Size')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    flops_per_epoch = [m['flops_per_epoch'] for m in all_metrics]
    plt.semilogx(batch_sizes, flops_per_epoch, '-o')
    plt.xlabel('Batch Size (log scale)')
    plt.ylabel('FLOPs per Epoch')
    plt.title('Computational Cost vs Batch Size')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('sgd_analysis.png')
    plt.close()
'''

'''
def initialize_mlp_params(input_size, hidden_size=10):
    """
    Initialize MLP parameters with Xavier/Glorot initialization

    params:
        input_size: Number of input features
        hidden_size: Number of hidden units

    return:
        Tuple of weights and biases (W1, b1, W2, b2)
    """
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / (hidden_size + 1))
    b2 = np.zeros(1)
    return W1, b1, W2, b2
'''
'''
def mlp_forward(X, W1, b1, W2, b2):
    """
    Forward pass through MLP

    params:
        X: Input features
        W1, b1: Hidden layer parameters
        W2, b2: Output layer parameters

    return:
        Tuple of (output probabilities, hidden activations, output pre-activation)
    """
    hidden_pre = np.dot(X, W1) + b1
    hidden = sigmoid(hidden_pre)
    output_pre = np.dot(hidden, W2) + b2
    output = sigmoid(output_pre)
    return output, hidden, output_pre
'''

'''
def mlp_backward(X, y, W2, output, hidden):
    """
    Backward pass through MLP using cross-entropy loss

    params:
        X: Input features
        y: True labels
        W2: Output layer weights
        output: Output probabilities
        hidden: Hidden layer activations

    return:
        Tuple of (gradients dictionary, loss value)
    """
    n_samples = X.shape[0]
    epsilon = 1e-15
    loss = -np.mean(y * np.log(output + epsilon) + (1 - y) * np.log(1 - output + epsilon))
    delta_output = output - y.reshape(-1, 1)
    delta_hidden = np.dot(delta_output, W2.T) * hidden * (1 - hidden)
    gradients = {
        'W1': np.dot(X.T, delta_hidden) / n_samples,
        'b1': np.mean(delta_hidden, axis=0),
        'W2': np.dot(hidden.T, delta_output) / n_samples,
        'b2': np.mean(delta_output, axis=0)
    }
    return gradients, loss
'''

'''
def train_mlp(X, y, learning_rate, num_epochs):
    """
    Train MLP model

    params:
        X: Training features
        y: Target values
        learning_rate: Learning rate
        num_epochs: Number of epochs

    return:
        Tuple of:
        - Dictionary of trained parameters
        - List of accuracies per epoch
        - List of losses per epoch
        - Dictionary of computational metrics
    """
    n_samples, n_features = X.shape
    W1, b1, W2, b2 = initialize_mlp_params(n_features)
    accuracies = []
    losses = []

    start_time = time.time()
    total_flops = 0

    for epoch in range(num_epochs):
        output, hidden, _ = mlp_forward(X, W1, b1, W2, b2)
        gradients, loss = mlp_backward(X, y, W2, output, hidden)
        W1 -= learning_rate * gradients['W1']
        b1 -= learning_rate * gradients['b1']
        W2 -= learning_rate * gradients['W2']
        b2 -= learning_rate * gradients['b2']
        predictions = (output >= 0.5).reshape(-1)
        accuracies.append(np.mean(predictions == y))
        losses.append(loss)
        epoch_flops = (
                2 * n_samples * n_features * 10 +
                2 * n_samples * 10 +
                2 * n_samples * 10 +
                n_samples +
                n_samples * 10 * 2 +
                n_features * 10 * 2
        )
        total_flops += epoch_flops

    wall_time = time.time() - start_time
    metrics = {
        'wall_time': wall_time,
        'total_flops': total_flops,
        'flops_per_epoch': total_flops / num_epochs
    }
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params, accuracies, losses, metrics
'''

'''
def compare_lr_vs_mlp(train_file, learning_rates, num_epochs):
    X, y = load_data(train_file)
    lr_results = []
    mlp_results = []

    with open('model_analysis.txt', 'w') as f:
        f.write("Model Comparison Analysis\n")
        f.write("=" * 50 + "\n\n")

        for lr in learning_rates:
            f.write(f"\nTraining with learning rate: {lr}\n")
            f.write("-" * 30 + "\n")
            weights, bias, lr_acc = train_logistic_regression(X, y, lr, num_epochs)
            lr_results.append({'learning_rate': lr, 'accuracies': lr_acc})
            _, mlp_acc, mlp_losses, mlp_metrics = train_mlp(X, y, lr, num_epochs)
            mlp_results.append(
                {'learning_rate': lr, 'accuracies': mlp_acc, 'losses': mlp_losses, 'metrics': mlp_metrics})
            f.write("\nEpoch-by-epoch comparison:\n")
            f.write(f"{'Epoch':>6} | {'LR Accuracy':>12} | {'MLP Accuracy':>12} | {'MLP Loss':>12}\n")
            f.write("-" * 50 + "\n")
            for epoch in range(num_epochs):
                f.write(
                    f"{epoch + 1:>6} | {lr_acc[epoch]:>12.4f} | {mlp_acc[epoch]:>12.4f} | {mlp_losses[epoch]:>12.4f}\n")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for result in lr_results:
        plt.plot(range(1, num_epochs + 1), result['accuracies'], '--', label=f'LR (lr={result["learning_rate"]})')
    for result in mlp_results:
        plt.plot(range(1, num_epochs + 1), result['accuracies'], '-', label=f'MLP (lr={result["learning_rate"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Convergence Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)

    plt.subplot(2, 2, 2)
    lr_final_acc = [r['accuracies'][-1] for r in lr_results]
    mlp_final_acc = [r['accuracies'][-1] for r in mlp_results]
    plt.semilogx(learning_rates, lr_final_acc, 'b--o', label='Logistic Regression')
    plt.semilogx(learning_rates, mlp_final_acc, 'r-o', label='MLP')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Final Accuracy')
    plt.title('Final Performance vs Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    mlp_times = [m['metrics']['wall_time'] for m in mlp_results]
    plt.semilogx(learning_rates, mlp_times, 'r-o', label='MLP')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Training Time (seconds)')
    plt.title('Computational Efficiency')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    for result in mlp_results:
        plt.plot(range(1, num_epochs + 1), result['losses'], label=f'lr={result["learning_rate"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('MLP Loss Convergence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
'''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("train_file", help="Path to training data")
    parser.add_argument("dev_file", help="Path to development data")
    parser.add_argument("num_epochs", type=int, help="Number of epochs")
    args = parser.parse_args()

    lrs = [3, 1.0, 0.1, 0.01, 0.001]
    plot_accuracy_vs_epoch(args.train_file, args.dev_file, lrs, args.num_epochs)
    #analyze_sgd_performance(args.train_file, 0.1, args.num_epochs, [1, 4, 16, 64, 256, len(load_data(args.train_file)[0])])
    #compare_lr_vs_mlp(args.train_file, lrs, args.num_epochs)
