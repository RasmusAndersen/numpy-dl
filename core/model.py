import numpy as np
from core.optimizer import SGD
from core.loss import CategoricalCrossEntropy
from core.utils import split_into_batches
from tqdm import tqdm


class Model(object):
    def __init__(self, net=None):
        self.net = [] if net is None else net

        self.loss_function = None
        self.optimizer = None

        self.model_compiled = False

    def add(self, function):
        self.net.append(function)

    def compile(self, optimizer=SGD(lr=0.01), loss=CategoricalCrossEntropy()):
        self.optimizer = optimizer
        self.loss_function = loss

        self.model_compiled = True

    def predict(self, x):
        intermediate = x
        for f in self.net:
            intermediate = f.forward(intermediate)
        return intermediate

    def backprop(self, ypred, ytarget):
        # compute the gradient of each layer
        next_grad = self.loss_function.derivative(ypred, ytarget)
        for i in range(len(self.net)-1, -1, -1): # TODO: convert to for layer in self.net[:-2:-1]
            next_grad = self.net[i].backward(next_grad)

        self.optimizer.update(self.net)

        return self.loss_function.loss(ypred, ytarget)

    def fit(self, x, y, epoch, batch_size=128):
        idx = np.arange(x.shape[0])
        for e in range(epoch):
            np.random.shuffle(idx)
            batch_idxs = split_into_batches(idx, batch_size)

            batch_loss, batch_acc = [], []
            for batch_idx in tqdm(batch_idxs):
                ytarget = y[batch_idx]
                ypred = self.predict(x[batch_idx])
                loss = self.backprop(ypred=ypred, ytarget=ytarget)
                batch_loss.append(loss)
                batch_acc.append(self.accuracy(ypred, ytarget))
            print(f'epoch {e}/{epoch}, loss: {np.mean(batch_loss)}, accuracy: {np.mean(batch_acc)}')

    def accuracy(self, ypred, ytarget):
        return np.mean(np.argmax(ypred, axis=1) == np.argmax(ytarget, axis=1))

    def evaluate(self, x, y):
        # TODO: implement evaluation
        raise NotImplementedError

    def save_weights(self):
        # TODO: implement saving weights
        raise NotImplementedError

    def load_weights(self):
        # TODO: implement loading weights
        raise NotImplementedError
