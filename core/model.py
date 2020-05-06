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

    def fit(self, x, y, x_val, y_val, epoch, batch_size=128):
        idx = np.arange(x.shape[0])
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        for e in range(epoch):
            np.random.shuffle(idx)
            batch_idxs = split_into_batches(idx, batch_size)

            epoch_loss, epoch_acc = [], []
            for batch_idx in tqdm(batch_idxs):
                ytarget = y[batch_idx]
                ypred = self.predict(x[batch_idx])
                batch_loss = self.backprop(ypred=ypred, ytarget=ytarget)
                epoch_loss.append(batch_loss)
                epoch_acc.append(self.accuracy(ypred, ytarget))

            train_loss.append(np.mean(epoch_loss))
            train_acc.append(np.mean(epoch_acc))

            ypred = self.predict(x_val)
            val_loss.append(np.mean(self.loss_function.loss(ypred, y_val)))
            val_acc.append(np.mean(self.accuracy(ypred, y_val)))
            print(f'epoch {e}/{epoch}, train loss: {train_loss[-1]}, train acc: {train_acc[-1]}, val loss: {val_loss[-1]}, val acc: {val_acc[-1]}')

        return train_loss, train_acc, val_loss, val_acc

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
