import numpy as np


class Initializer(object):
    pass


class Zeros(Initializer):
    def __call__(self, shape):
        return np.zeros(shape)


class Normal(Initializer):
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self. scale = scale

    def __call__(self, shape):
        return np.random.normal(size=shape, loc=self.loc, scale=self.scale)


class GlorotNormal(Initializer):
    def __call__(self, shape):
        return Normal(loc=0, scale=np.sqrt(2 / np.sum(shape)))(shape)
