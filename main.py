import numpy as np
import sys
# import matplotlib.pyplot as plt
# import time

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
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    accuracies = []
    
    for epoch in range(num_epochs):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)
        
        dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1/n_samples) * np.sum(y_predicted - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        accuracy = compute_accuracy(X, y, weights, bias)
        accuracies.append(accuracy)
    
    return weights, bias, accuracies

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
    y = data[:, -1]   # Last column
    return X, y

# def plot_accuracy_vs_epoch(train_file, learning_rates, num_epochs, num_runs=5):
#     """
#     Plot accuracy vs epoch for different learning rates
    
#     params:
#         train_file (str): Path to training data
#         learning_rates (list): List of learning rates to try
#         num_epochs (int): Number of epochs to train
#         num_runs (int): Number of runs for each learning rate
#     """
#     X, y = load_data(train_file)
#     plt.figure(figsize=(10, 6))
    
#     for lr in learning_rates:
#         all_accuracies = []
#         for _ in range(num_runs):
#             _, _, accuracies = train_logistic_regression(X, y, lr, num_epochs)
#             all_accuracies.append(accuracies)
        
#         mean_accuracies = np.mean(all_accuracies, axis=0)
#         std_accuracies = np.std(all_accuracies, axis=0)
#         epochs = np.arange(1, num_epochs + 1)
        
#         plt.plot(epochs, mean_accuracies, label=f'lr={lr}')
#         plt.fill_between(epochs, 
#                         mean_accuracies - std_accuracies,
#                         mean_accuracies + std_accuracies,
#                         alpha=0.2)
    
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs Epoch for Different Learning Rates')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('accuracy_vs_epoch.png')
#     plt.close()

# def train_sgd_logistic_regression(X, y, learning_rate, num_epochs, batch_size):
#     n_samples, n_features = X.shape
#     weights = np.zeros(n_features)
#     bias = 0
#     accuracies = []
    
#     start_time = time.time()
#     total_flops = 0
    
#     for epoch in range(num_epochs):
#         indices = np.random.permutation(n_samples)
#         X_shuffled = X[indices]
#         y_shuffled = y[indices]
        
#         for i in range(0, n_samples, batch_size):
#             batch_X = X_shuffled[i:i + batch_size]
#             batch_y = y_shuffled[i:i + batch_size]
#             batch_size_actual = len(batch_X)
            
#             linear_model = np.dot(batch_X, weights) + bias
#             y_predicted = sigmoid(linear_model)
            
#             dw = (1/batch_size_actual) * np.dot(batch_X.T, (y_predicted - batch_y))
#             db = (1/batch_size_actual) * np.sum(y_predicted - batch_y)
            
#             weights -= learning_rate * dw
#             bias -= learning_rate * db
            
#             total_flops += batch_size_actual * (4 * n_features + 2)
        
#         accuracy = compute_accuracy(X, y, weights, bias)
#         accuracies.append(accuracy)
    
#     wall_time = time.time() - start_time
#     metrics = {
#         'wall_time': wall_time,
#         'total_flops': total_flops,
#         'flops_per_epoch': total_flops / num_epochs
#     }
    
#     return weights, bias, accuracies, metrics

# def analyze_sgd_performance(train_file, learning_rate, num_epochs, 
#                           batch_sizes):
#     """
#     Analyze SGD performance for different batch sizes
    
#     params:
#         train_file: Path to training data
#         learning_rate: Learning rate
#         num_epochs: Number of epochs
#         batch_sizes: List of batch sizes to try
#     """
#     X, y = load_data(train_file)
#     n_samples = len(X)
    
#     # Metrics storage
#     all_metrics = []
#     all_accuracies = []
    
#     # Create analysis file
#     with open('sgd_analysis.txt', 'w') as f:
#         f.write("SGD Performance Analysis\n")
#         f.write("=" * 50 + "\n\n")
        
#         # Training with different batch sizes
#         for batch_size in batch_sizes:
#             f.write(f"\nTraining with batch size: {batch_size}\n")
#             f.write("-" * 30 + "\n")
            
#             weights, bias, accuracies, metrics = train_sgd_logistic_regression(
#                 X, y, learning_rate, num_epochs, batch_size
#             )
            
#             all_metrics.append({
#                 'batch_size': batch_size,
#                 **metrics
#             })
#             all_accuracies.append(accuracies)
            
#             # Write epoch-by-epoch results
#             f.write("\nEpoch-by-epoch accuracy:\n")
#             f.write(f"{'Epoch':>6} | {'Accuracy':>12}\n")
#             f.write("-" * 25 + "\n")
#             for epoch, acc in enumerate(accuracies, 1):
#                 f.write(f"{epoch:>6} | {acc:>12.4f}\n")
    
#     plt.figure(figsize=(15, 5))
    
#     # Plot 1: Accuracy vs Epoch for different batch sizes
#     plt.subplot(1, 3, 1)
#     for i, batch_size in enumerate(batch_sizes):
#         plt.plot(range(1, num_epochs + 1), all_accuracies[i], 
#                 label=f'Batch Size={batch_size}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Convergence by Batch Size')
#     plt.legend()
#     plt.grid(True)
    
#     # Plot 2: Wall Time vs Batch Size
#     plt.subplot(1, 3, 2)
#     wall_times = [m['wall_time'] for m in all_metrics]
#     plt.semilogx(batch_sizes, wall_times, '-o')
#     plt.xlabel('Batch Size (log scale)')
#     plt.ylabel('Wall Time (seconds)')
#     plt.title('Wall Time vs Batch Size')
#     plt.grid(True)
    
#     # Plot 3: FLOPs per Epoch vs Batch Size
#     plt.subplot(1, 3, 3)
#     flops_per_epoch = [m['flops_per_epoch'] for m in all_metrics]
#     plt.semilogx(batch_sizes, flops_per_epoch, '-o')
#     plt.xlabel('Batch Size (log scale)')
#     plt.ylabel('FLOPs per Epoch')
#     plt.title('Computational Cost vs Batch Size')
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig('sgd_analysis.png')
#     plt.close()
    
#     # Write summary statistics to file
#     with open('sgd_analysis.txt', 'a') as f:
#         f.write("\n\nSummary Statistics\n")
#         f.write("=" * 50 + "\n\n")
#         f.write(f"{'Batch Size':^12} | {'Wall Time (s)':^15} | {'FLOPs/Epoch':^15} | {'Total FLOPs':^15}\n")
#         f.write("-" * 80 + "\n")
#         for metrics in all_metrics:
#             f.write(f"{metrics['batch_size']:^12} | {metrics['wall_time']:^15.3f} | "
#                    f"{metrics['flops_per_epoch']:^15.2e} | {metrics['total_flops']:^15.2e}\n")

# def initialize_mlp_params(input_size, hidden_size=10):
#     """
#     Initialize MLP parameters with Xavier/Glorot initialization
    
#     params:
#         input_size: Number of input features
#         hidden_size: Number of hidden units
        
#     return:
#         Tuple of weights and biases (W1, b1, W2, b2)
#     """
#     W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
#     b1 = np.zeros(hidden_size)
#     W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / (hidden_size + 1))
#     b2 = np.zeros(1)
#     return W1, b1, W2, b2

# def mlp_forward(X, W1, b1, 
#                 W2, b2):
#     """
#     Forward pass through MLP
    
#     params:
#         X: Input features
#         W1, b1: Hidden layer parameters
#         W2, b2: Output layer parameters
        
#     return:
#         Tuple of (output probabilities, hidden activations, output pre-activation)
#     """
#     hidden_pre = np.dot(X, W1) + b1
#     hidden = sigmoid(hidden_pre)
#     output_pre = np.dot(hidden, W2) + b2
#     output = sigmoid(output_pre)
#     return output, hidden, output_pre

# def mlp_backward(X, y, W2, 
#                 output, hidden):
#     """
#     Backward pass through MLP using cross-entropy loss
    
#     params:
#         X: Input features
#         y: True labels
#         W2: Output layer weights
#         output: Output probabilities
#         hidden: Hidden layer activations
        
#     return:
#         Tuple of (gradients dictionary, loss value)
#     """
#     n_samples = X.shape[0]
    
#     # Compute cross-entropy loss
#     epsilon = 1e-15
#     loss = -np.mean(y * np.log(output + epsilon) + (1 - y) * np.log(1 - output + epsilon))
    
#     # Output layer gradients
#     delta_output = output - y.reshape(-1, 1)
    
#     # Hidden layer gradients
#     delta_hidden = np.dot(delta_output, W2.T) * hidden * (1 - hidden)
    
#     # Compute gradients
#     gradients = {
#         'W1': np.dot(X.T, delta_hidden) / n_samples,
#         'b1': np.mean(delta_hidden, axis=0),
#         'W2': np.dot(hidden.T, delta_output) / n_samples,
#         'b2': np.mean(delta_output, axis=0)
#     }
    
#     return gradients, loss

# def train_mlp(X, y, learning_rate, 
#               num_epochs):
#     """
#     Train MLP model
    
#     params:
#         X: Training features
#         y: Target values
#         learning_rate: Learning rate
#         num_epochs: Number of epochs
        
#     return:
#         Tuple of:
#         - Dictionary of trained parameters
#         - List of accuracies per epoch
#         - List of losses per epoch
#         - Dictionary of computational metrics
#     """
#     n_samples, n_features = X.shape
#     W1, b1, W2, b2 = initialize_mlp_params(n_features)
#     accuracies = []
#     losses = []
    
#     # Metrics tracking
#     start_time = time.time()
#     total_flops = 0
    
#     for epoch in range(num_epochs):
#         # Forward pass
#         output, hidden, _ = mlp_forward(X, W1, b1, W2, b2)
        
#         # Backward pass
#         gradients, loss = mlp_backward(X, y, W2, output, hidden)
        
#         # Update parameters
#         W1 -= learning_rate * gradients['W1']
#         b1 -= learning_rate * gradients['b1']
#         W2 -= learning_rate * gradients['W2']
#         b2 -= learning_rate * gradients['b2']
        
#         # Track metrics
#         predictions = (output >= 0.5).reshape(-1)
#         accuracy = np.mean(predictions == y)
#         accuracies.append(accuracy)
#         losses.append(loss)
        
#         # Track FLOPs
#         epoch_flops = (
#             2 * n_samples * n_features * 10 +  # Hidden layer forward
#             2 * n_samples * 10 +  # Hidden layer activation
#             2 * n_samples * 10 +  # Output layer forward
#             n_samples +  # Output layer activation
#             n_samples * 10 * 2 +  # Backprop through hidden layer
#             n_features * 10 * 2  # Weight updates
#         )
#         total_flops += epoch_flops
    
#     wall_time = time.time() - start_time
#     metrics = {
#         'wall_time': wall_time,
#         'total_flops': total_flops,
#         'flops_per_epoch': total_flops / num_epochs
#     }
    
#     params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
#     return params, accuracies, losses, metrics

# def compare_lr_vs_mlp(train_file, learning_rates, num_epochs):
#     X, y = load_data(train_file)
#     lr_results = []
#     mlp_results = []
    
#     with open('model_analysis.txt', 'w') as f:
#         f.write("Model Comparison Analysis\n")
#         f.write("=" * 50 + "\n\n")
        
#         for lr in learning_rates:
#             f.write(f"\nTraining with learning rate: {lr}\n")
#             f.write("-" * 30 + "\n")
            
#             # Logistic Regression
#             weights, bias, lr_acc = train_logistic_regression(X, y, lr, num_epochs)
#             lr_results.append({
#                 'learning_rate': lr,
#                 'accuracies': lr_acc
#             })
            
#             # MLP
#             _, mlp_acc, mlp_losses, mlp_metrics = train_mlp(X, y, lr, num_epochs)
#             mlp_results.append({
#                 'learning_rate': lr,
#                 'accuracies': mlp_acc,
#                 'losses': mlp_losses,
#                 'metrics': mlp_metrics
#             })
            
#             f.write("\nEpoch-by-epoch comparison:\n")
#             f.write(f"{'Epoch':>6} | {'LR Accuracy':>12} | {'MLP Accuracy':>12} | {'MLP Loss':>12}\n")
#             f.write("-" * 50 + "\n")
#             for epoch in range(num_epochs):
#                 f.write(f"{epoch+1:>6} | {lr_acc[epoch]:>12.4f} | {mlp_acc[epoch]:>12.4f} | {mlp_losses[epoch]:>12.4f}\n")
    
#     plt.figure(figsize=(15, 10))
    
#     plt.subplot(2, 2, 1)
#     for result in lr_results:
#         plt.plot(range(1, num_epochs + 1), result['accuracies'], 
#                 '--', label=f'LR (lr={result["learning_rate"]})')
#     for result in mlp_results:
#         plt.plot(range(1, num_epochs + 1), result['accuracies'], 
#                 '-', label=f'MLP (lr={result["learning_rate"]})')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Model Convergence Comparison')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
    
#     plt.subplot(2, 2, 2)
#     lr_final_acc = [result['accuracies'][-1] for result in lr_results]
#     mlp_final_acc = [result['accuracies'][-1] for result in mlp_results]
#     plt.semilogx(learning_rates, lr_final_acc, 'b--o', label='Logistic Regression')
#     plt.semilogx(learning_rates, mlp_final_acc, 'r-o', label='MLP')
#     plt.xlabel('Learning Rate (log scale)')
#     plt.ylabel('Final Accuracy')
#     plt.title('Final Performance vs Learning Rate')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(2, 2, 3)
#     mlp_times = [result['metrics']['wall_time'] for result in mlp_results]
#     plt.semilogx(learning_rates, mlp_times, 'r-o', label='MLP')
#     plt.xlabel('Learning Rate (log scale)')
#     plt.ylabel('Training Time (seconds)')
#     plt.title('Computational Efficiency')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(2, 2, 4)
#     for result in mlp_results:
#         plt.plot(range(1, num_epochs + 1), result['losses'], 
#                 label=f'lr={result["learning_rate"]}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Cross-Entropy Loss')
#     plt.title('MLP Loss Convergence')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig('model_comparison.png', bbox_inches='tight')
#     plt.close()
    
#     with open('model_analysis.txt', 'a') as f:
#         f.write("\n\nSummary Statistics\n")
#         f.write("=" * 50 + "\n\n")
#         f.write(f"{'Learning Rate':^12} | {'LR Accuracy':^12} | {'MLP Accuracy':^12} | "
#                 f"{'MLP Time (s)':^12} | {'MLP FLOPs':^15}\n")
#         f.write("-" * 80 + "\n")
        
#         for lr_res, mlp_res in zip(lr_results, mlp_results):
#             lr = lr_res['learning_rate']
#             f.write(f"{lr:^12.3e} | {lr_res['accuracies'][-1]:^12.4f} | "
#                    f"{mlp_res['accuracies'][-1]:^12.4f} | {mlp_res['metrics']['wall_time']:^12.3f} | "
#                    f"{mlp_res['metrics']['total_flops']:^15.2e}\n")

def main():
    # Get command line arguments
    train_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    num_epochs = int(sys.argv[3])
    
    # Load training data and train model
    X, y = load_data(train_file)
    weights, bias, _ = train_logistic_regression(X, y, learning_rate, num_epochs)
    
    # Print parameters (weights followed by bias) on a single line
    print(" ".join(map(str, np.concatenate([weights, [bias]]))))
    
    # # Generate accuracy plots and analysis
    # learning_rates = [0.001, 0.01, 0.1, 1.0, 10.0]
    # plot_accuracy_vs_epoch(train_file, learning_rates, num_epochs)
    
    # # Analyze SGD performance and save to file
    # batch_sizes = [1, 4, 16, 64, 256, len(X)]  # From single sample to full batch
    # analyze_sgd_performance(train_file, learning_rate, num_epochs, batch_sizes)
    
    # # Compare Logistic Regression vs MLP and save to file
    # compare_lr_vs_mlp(train_file, learning_rates, num_epochs)

if __name__ == "__main__":
    main()
