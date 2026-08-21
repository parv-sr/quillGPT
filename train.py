import numpy as np

from typing import Tuple

x = np.array([np.random.random() for x in range(3)])

w1 = np.array(
    [
    [np.random.random() for x in range(3)],
    [np.random.random() for x in range(3)],
    [np.random.random() for x in range(3)],
    [np.random.random() for x in range(3)]
    ]
)

b1 = np.array([np.random.random() for x in range(4)])

z1 = x @ w1.T + b1

a1 = 1 / (1 + np.exp(-z1))

w2 = np.array(
    [
    [np.random.random() for x in range(4)],
    [np.random.random() for x in range(4)],
    [np.random.random() for x in range(4)],
    ]
)

b2 = np.array([np.random.random() for x in range(3)])

z2 = a1 @ w2.T + b2

a2 = np.tanh(z2)

print(a2)


class Neuron:
    def __init__(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights
        self.baises = biases
