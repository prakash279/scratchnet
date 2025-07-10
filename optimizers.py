import numpy as np

class Optimizer:
    def update(self, layer):
        raise NotImplementedError("Bro implement the update method.")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.0001):
        self.lr = learning_rate

    def update(self, layer):
        layer.weights -= self.lr * layer.grad_weights
        layer.biases -= self.lr * layer.grad_biases

class SGDWithMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities_w = {}
        self.velocities_b = {}

    def update(self, layer):
        if id(layer) not in self.velocities_w:
            self.velocities_w[id(layer)] = np.zeros_like(layer.weights)
            self.velocities_b[id(layer)] = np.zeros_like(layer.biases)

        v_w = self.velocities_w[id(layer)]
        v_b = self.velocities_b[id(layer)]

        v_w = self.momentum * v_w - self.lr * layer.grad_weights
        v_b = self.momentum * v_b - self.lr * layer.grad_biases

        self.velocities_w[id(layer)] = v_w
        self.velocities_b[id(layer)] = v_b

        layer.weights += v_w
        layer.biases += v_b

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self.m:
            self.m[lid] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            self.v[lid] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            self.t[lid] = 0

        self.t[lid] += 1

        m_w = self.m[lid]['weights']
        m_b = self.m[lid]['biases']
        v_w = self.v[lid]['weights']
        v_b = self.v[lid]['biases']

        # Update biased moments
        m_w = self.beta1 * m_w + (1 - self.beta1) * layer.grad_weights
        m_b = self.beta1 * m_b + (1 - self.beta1) * layer.grad_biases
        v_w = self.beta2 * v_w + (1 - self.beta2) * (layer.grad_weights ** 2)
        v_b = self.beta2 * v_b + (1 - self.beta2) * (layer.grad_biases ** 2)

        # Bias correction
        m_w_hat = m_w / (1 - self.beta1 ** self.t[lid])
        m_b_hat = m_b / (1 - self.beta1 ** self.t[lid])
        v_w_hat = v_w / (1 - self.beta2 ** self.t[lid])
        v_b_hat = v_b / (1 - self.beta2 ** self.t[lid])

        # Parameter update
        layer.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.biases -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        # Save state
        self.m[lid]['weights'] = m_w
        self.m[lid]['biases'] = m_b
        self.v[lid]['weights'] = v_w
        self.v[lid]['biases'] = v_b

# 🎛 Optimizer Factory / Wrapper
class OptimizerFactory:
    @staticmethod
    def get(optimizer_name, **kwargs):
        name = optimizer_name.lower()
        if name == "sgd":
            return SGD(**kwargs)
        elif name == "momentum":
            return SGDWithMomentum(**kwargs)
        elif name == "adam":
            return Adam(**kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
