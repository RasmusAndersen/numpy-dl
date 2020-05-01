from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def split_into_batches(idx, batch_size):
    # split a list of indexes into max possible _whole_ batches
    idx_at_last_whole_batch = int(np.floor(idx.shape[0] / batch_size) * batch_size)
    whole_batches = idx[:idx_at_last_whole_batch]
    nb_whole_batches = np.floor(idx.shape[0] / batch_size)
    batches = np.array_split(whole_batches, nb_whole_batches)

    # add the leftover as the last batch (i.e. it will have a size < batch_size)
    if idx[idx_at_last_whole_batch:]:
        batches.append(idx[idx_at_last_whole_batch:])

    return batches
