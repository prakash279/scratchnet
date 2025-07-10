import numpy as np
import pickle
import os

def split_data(X, y, test_ratio=0.2, shuffle=True):
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

    test_size = int(X.shape[0] * test_ratio)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    return X_train, X_test, y_train, y_test

def one_hot(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1
    onehot = np.zeros((y.shape[0], num_classes))
    onehot[np.arange(y.shape[0]), y] = 1
    return onehot

def accuracy(y_pred, y_true):
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    else:
        y_pred_labels = (y_pred > 0.5).astype(int).ravel()
        y_true_labels = y_true.ravel()
    return np.mean(y_pred_labels == y_true_labels)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_epoch(epoch, loss, acc):
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

def save_model(model, filepath="model.pkl"):
    state = []
    for layer in model:
        layer_state = {}
        if hasattr(layer, 'weights'):
            layer_state['weights'] = layer.weights
        if hasattr(layer, 'biases'):
            layer_state['biases'] = layer.biases
        state.append(layer_state)

    with open(filepath, "wb") as f:
        pickle.dump(state, f)
    print(f"[💾] Model saved to {filepath}")

def load_model(model, filepath="model.pkl"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")

    with open(filepath, "rb") as f:
        state = pickle.load(f)

    for layer, layer_state in zip(model, state):
        if hasattr(layer, 'weights') and 'weights' in layer_state:
            layer.weights = layer_state['weights']
        if hasattr(layer, 'biases') and 'biases' in layer_state:
            layer.biases = layer_state['biases']

    print(f"[📂] Model loaded from {filepath}")
