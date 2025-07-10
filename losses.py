import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError("Loss forward not implemented.")

    def backward(self):
        raise NotImplementedError("Loss backward not implemented.")

class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def backward(self):
        # dL/dy_pred
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]

class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        """
        y_pred: output probabilities (after softmax or sigmoid)
        y_true: one-hot vectors or labels (0/1) or class indices
        """
        self.y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # prevent log(0)
        self.y_true = y_true

        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # Binary classification
            self.binary = True
            loss = -np.mean(y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        else:
            # Multiclass classification with one-hot labels
            self.binary = False
            loss = -np.mean(np.sum(y_true * np.log(self.y_pred), axis=1))
        
        return loss

    def backward(self):
        # Gradient of binary cross-entropy or categorical cross-entropy
        if self.binary:
            return (self.y_pred - self.y_true) / self.y_true.shape[0]
        else:
            return (self.y_pred - self.y_true) / self.y_true.shape[0]
