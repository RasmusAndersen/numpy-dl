import numpy as np
from core.initializer import GlorotNormal, Normal, Zeros


class Layer(object):
    def __init__(self, shape=None, name=None):
        self.name = None
        self.last_input = None
        self.trainable = False

    def get_name(self):
        return self.name

    def forward(self, x):
        raise NotImplementedError

    def backward(self, prev_delta, activation):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, shape, name=None, initializer=Normal(loc=0, scale=0.01)):
        self.initializer = initializer
        self.weights = self.initializer(shape)
        self.bias = Zeros()(self.weights.shape[-1])
        super(Dense, self).__init__(name=name)

        self.trainable = True
        self.weights_grad = None
        self.bias_grad = None

    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, prev_delta, activation):
        # nice explanation: https://youtu.be/tIeHLnjs5U8?t=281
        activation_gradient = activation.gradient() * prev_delta
        self.weights_grad = np.dot(self.last_input.T, activation_gradient)
        self.bias_grad = np.mean(activation_gradient, axis=0)
        return np.dot(activation_gradient, self.weights.T)


class Dropout(Layer):
    def __init__(self, prob, name=None):
        assert (prob > 0.) and (prob < 1.)
        self.prob = prob
        super(Dropout, self).__init__(name=name)

    def forward(self, x, training=False):
        mask = np.random.random(x.shape) > self.prob
        self.last_input = mask
        return x * mask

    def backward(self, prev_delta, activation):
        return prev_delta * self.last_input
