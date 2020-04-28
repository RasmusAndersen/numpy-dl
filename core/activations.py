import numpy as np


class Activation(object):
    def __init__(self, name=None):
        self.name = name
        self.last_input = None
        self.trainable = False

    def forward(self, x, gradient=False):
        raise NotImplementedError

    def gradient(self, x=None):
        raise NotImplementedError


class ReLU(Activation):
    def __init__(self, name=None):
        super(ReLU, self).__init__(name=name)

    def forward(self, x, gradient=False):
        self.last_input = x
        if gradient:
            return self.gradient(x)
        return np.maximum(0.0, x)

    def gradient(self, x=None):
        if x is None:
            return (self.last_input > 0).astype(np.float64)
        return (x > 0).astype(np.float64)


class Softmax(Activation):
    def __init__(self, name=None):
        super(Softmax, self).__init__(name=name)

    def forward(self, x, gradient=False):
        self.last_input = x
        if gradient:
            return self.gradient(x)

        m = np.atleast_2d(x.max(axis=1)).T
        return (np.exp(x - m).T / np.sum(np.exp(x - m).T, axis=0)).T

    def gradient(self, x=None):
        if x is None:
            return np.ones_like(self.last_input)
        else:
            return np.ones_like(x)
