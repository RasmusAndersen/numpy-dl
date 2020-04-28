import numpy as np


class Optimizer(object):
    def __init__(self, lr, decay=0.0, lr_max=np.inf, lr_min=0):
        self.lr = lr
        # TODO: implement decaying learning rate


class SGD(Optimizer):
    def __init__(self, lr, decay=0.0, moment=0):
        super(SGD, self).__init__(lr, decay)
        # TODO: implement moment

    def update(self, net):
        for layer in net:
            if layer.trainable:
                layer.weights -= self.lr * layer.weights_grad
                layer.bias -= self.lr * layer.bias_grad
