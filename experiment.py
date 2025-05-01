import numpy as np
import matplotlib.pyplot as plt
import time
from main import train_logistic_regression, compute_accuracy, sigmoid
import os

def load_data(path):
    data = np.loadtxt(path)
    return data[:, :-1], data[:, -1].astype(int)

# Part 1: Batch Gradient Descent Analysis (Extra Credit Part 1)
def plot_accuracy_vs_epoch(train_file, dev_file, learning_rates, num_epochs, num_runs=5):
    X_train, y_train = load_data(train_file)
    X_dev, y_dev = load_data(dev_file)

    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, num_epochs + 1)

    for lr in learning_rates:
        train_runs, dev_runs = [], []
        for _ in range(num_runs):
            w, b, tr_hist, dv_hist = train_logistic_regression(
                X_train, y_train, lr, num_epochs, X_dev, y_dev
            )
            train_runs.append(tr_hist)
            dev_runs.append(dv_hist)

        train_arr = np.array(train_runs)
        dev_arr = np.array(dev_runs)

        # compute stats
        train_mean = train_arr.mean(axis=0)
        train_std = train_arr.std(axis=0)
        train_min = train_arr.min(axis=0)
        train_max = train_arr.max(axis=0)

        dev_mean = dev_arr.mean(axis=0)
        dev_std = dev_arr.std(axis=0)
        dev_min = dev_arr.min(axis=0)
        dev_max = dev_arr.max(axis=0)

        # plot train: dashed mean, then ribbons
        plt.plot(epochs, train_mean, '--', label=f"train LR={lr}")
        plt.fill_between(epochs, train_min, train_max, alpha=0.1)
        plt.fill_between(epochs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.2)
        # plot dev: solid mean, then ribbons
        plt.plot(epochs, dev_mean, '-', label=f"dev LR={lr}")
        plt.fill_between(epochs, dev_min, dev_max, alpha=0.1)
        plt.fill_between(epochs,
                        dev_mean - dev_std,
                        dev_mean + dev_std,
                        alpha=0.07)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Batch GD: Train vs Dev Accuracy by Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/accuracy_vs_epoch.png")
    plt.close()

# Part 2: Stochastic Gradient Descent Analysis (Extra Credit)
def train_sgd_logistic_regression(X, y, learning_rate, num_epochs, batch_size):
    n_samples, n_features = X.shape
    weights = np.random.uniform(-0.01, 0.01, size=n_features)
    bias = np.random.uniform(-0.01, 0.01)

    accuracies = []
    total_flops = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        rand = np.random.permutation(n_samples)
        X_shuf, y_shuf = X[rand], y[rand]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]
            m = len(X_batch)

            z = X_batch.dot(weights) + bias
            y_pred = sigmoid(z)
            dw = X_batch.T.dot(y_pred - y_batch) / m
            db = (y_pred - y_batch).mean()
            weights -= learning_rate * dw
            bias -= learning_rate * db

            total_flops += m * (4 * n_features + 2)

        accuracies.append(compute_accuracy(X, y, weights, bias))

    wall_time = time.time() - start_time
    metrics = {
        'wall_time': wall_time,
        'total_flops': total_flops,
        'flops_per_epoch': total_flops / num_epochs
    }
    return weights, bias, accuracies, metrics

def analyze_sgd_performance(train_file, learning_rate, num_epochs, batch_sizes):
    X, y = load_data(train_file)
    all_metrics = []
    all_accs = []

    for bs in batch_sizes:
        _, _, accs, metrics = train_sgd_logistic_regression(
            X, y, learning_rate, num_epochs, bs
        )
        all_accs.append(accs)
        all_metrics.append({'batch_size': bs, **metrics})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convergence curves
    for idx, bs in enumerate(batch_sizes):
        axes[0].plot(range(1, num_epochs+1), all_accs[idx], label=f'BS={bs}')
    axes[0].set(title='SGD: Accuracy vs Epoch', xlabel='Epoch', ylabel='Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Wall time vs batch size
    wall_times = [m['wall_time'] for m in all_metrics]
    axes[1].semilogx(batch_sizes, wall_times, '-o')
    axes[1].set(title='Wall Time vs Batch Size', xlabel='Batch Size', ylabel='Seconds')
    axes[1].grid(True)

    # FLOPs per epoch vs batch size
    flops = [m['flops_per_epoch'] for m in all_metrics]
    axes[2].semilogx(batch_sizes, flops, '-o')
    axes[2].set(title='FLOPs/Epoch vs Batch Size', xlabel='Batch Size', ylabel='FLOPs')
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig('output/sgd_analysis.png')
    plt.close(fig)

# Part 3: Multi-Layer Perceptron Comparison (Extra Credit)
def initialize_mlp_params(input_size, hidden_size=10):
    w1 = np.random.uniform(-0.01, 0.01, size=(input_size, hidden_size))
    b1 = np.random.uniform(-0.01, 0.01, size=hidden_size)
    w2 = np.random.uniform(-0.01, 0.01, size=(hidden_size, 1))
    b2 = np.random.uniform(-0.01, 0.01, size=1)
    return w1, b1, w2, b2

def mlp_forward(X, W1, b1, W2, b2):
    z1 = X.dot(W1) + b1
    h1 = sigmoid(z1)
    z2 = h1.dot(W2) + b2
    out = sigmoid(z2)
    return out, h1

def mlp_backward(X, y, h1, out, W2):
    n = X.shape[0]
    delta2 = out - y.reshape(-1, 1)
    grad_W2 = h1.T.dot(delta2) / n
    grad_b2 = delta2.mean(axis=0)
    delta1 = (delta2.dot(W2.T)) * (h1 * (1 - h1))
    grad_W1 = X.T.dot(delta1) / n
    grad_b1 = delta1.mean(axis=0)
    return grad_W1, grad_b1, grad_W2, grad_b2

def train_mlp(X, y, learning_rate, num_epochs):
    n_samples, n_features = X.shape
    W1, b1, W2, b2 = initialize_mlp_params(n_features)

    accuracies = []
    losses = []
    total_flops = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        out, h1 = mlp_forward(X, W1, b1, W2, b2)
        eps = 1e-15
        loss = -np.mean(y.reshape(-1,1)*np.log(out+eps) + (1-y).reshape(-1,1)*np.log(1-out+eps))
        losses.append(loss)
        gW1, gb1, gW2, gb2 = mlp_backward(X, y, h1, out, W2)
        W1 -= learning_rate * gW1
        b1 -= learning_rate * gb1
        W2 -= learning_rate * gW2
        b2 -= learning_rate * gb2
        preds = (out >= 0.5).astype(int).reshape(-1)
        accuracies.append(np.mean(preds == y))

        hidden_size = h1.shape[1]
        ops_forward = 2 * n_samples * (n_features * hidden_size + hidden_size)
        ops_activate = 2 * n_samples * (hidden_size + 1)
        ops_backward = 2 * n_samples * (n_features * hidden_size + hidden_size * 1)
        total_flops += ops_forward + ops_activate + ops_backward

    wall_time = time.time() - start_time
    metrics = {
        'wall_time': wall_time,
        'total_flops': total_flops,
        'flops_per_epoch': total_flops / num_epochs
    }
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, accuracies, losses, metrics

def compare_lr_vs_mlp(train_file, dev_file, learning_rates, num_epochs, num_runs=5):
    """Compare logistic regression and MLP performance across learning rates."""
    X_train, y_train = load_data(train_file)
    X_dev, y_dev = load_data(dev_file)

    # Create subplot grid: one row per learning rate
    fig, axes = plt.subplots(len(learning_rates), 1, figsize=(12, 4*len(learning_rates)))
    epochs = np.arange(1, num_epochs + 1)

    for i, lr in enumerate(learning_rates):
        lr_train_runs = []
        lr_dev_runs = []
        mlp_train_runs = []
        mlp_dev_runs = []
        
        for _ in range(num_runs):
            # Train logistic regression
            _, _, tr_hist, dv_hist = train_logistic_regression(
                X_train, y_train, lr, num_epochs, X_dev, y_dev
            )
            lr_train_runs.append(tr_hist)
            lr_dev_runs.append(dv_hist)
            
            # Train MLP
            params, train_acc, _, _ = train_mlp(X_train, y_train, lr, num_epochs)
            mlp_train_runs.append(train_acc)
            
            # Compute MLP dev accuracy
            dev_acc = []
            W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
            for epoch in range(num_epochs):
                out, _ = mlp_forward(X_dev, W1, b1, W2, b2)
                preds = (out >= 0.5).astype(int).reshape(-1)
                dev_acc.append(np.mean(preds == y_dev))
            mlp_dev_runs.append(dev_acc)

        # Compute mean curves
        lr_train_mean = np.mean(lr_train_runs, axis=0)
        lr_dev_mean = np.mean(lr_dev_runs, axis=0)
        mlp_train_mean = np.mean(mlp_train_runs, axis=0)
        mlp_dev_mean = np.mean(mlp_dev_runs, axis=0)

        # Plot accuracies for this learning rate
        ax = axes[i]
        ax.plot(epochs, lr_train_mean, '--', label='LogReg Train', color='blue')
        ax.plot(epochs, lr_dev_mean, ':', label='LogReg Dev', color='blue')
        ax.plot(epochs, mlp_train_mean, '-', label='MLP Train', color='red')
        ax.plot(epochs, mlp_dev_mean, '-.', label='MLP Dev', color='red')
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Rate = {lr}")
        ax.legend()
        ax.grid(True)
        
        # Add y-axis limits to better see differences
        ax.set_ylim(0.4, 1.0)

    plt.suptitle("Model Comparison: Training and Development Set Accuracies", y=1.02)
    plt.tight_layout()
    plt.savefig('output/model_comparison.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    TRAIN_FILE = 'data/train.txt'
    DEV_FILE = 'data/dev.txt'
    EPOCHS = 100
    LEARNING_RATES = [3, 1.0, 0.1, 0.01, 0.001]
    NUM_RUNS = 5
    BATCH_SIZES = [1, 4, 16, 64, 256, None]  # none just means wree using the full batch


    os.makedirs('output', exist_ok=True)

    # Run experiments:
    # 1. Batch GD analysis
    plot_accuracy_vs_epoch(TRAIN_FILE, DEV_FILE, LEARNING_RATES, EPOCHS, NUM_RUNS)
    
    # 2. SGD analysis
    X_train, _ = load_data(TRAIN_FILE)
    bs_list = [bs if bs is not None else X_train.shape[0] for bs in BATCH_SIZES]
    analyze_sgd_performance(TRAIN_FILE, 0.1, EPOCHS, bs_list)
    
    # 3. MLP comparison
    compare_lr_vs_mlp(TRAIN_FILE, DEV_FILE, LEARNING_RATES, EPOCHS, NUM_RUNS)
