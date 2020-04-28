import numpy as np


class Loss(object):
    pass


class CategoricalCrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, ypred, ytarget):
        return -np.sum(ytarget * np.log(ypred + 1e-10)) / np.atleast_1d(ypred).shape[0]

    def derivative(self, ypred, ytarget):
        return ypred - ytarget
