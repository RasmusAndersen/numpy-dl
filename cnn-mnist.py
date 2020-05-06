import numpy as np
from core.layers import Dense, Dropout, Conv2D, Flatten, ReLU, Softmax, MaxPooling2D
from core.model import Model
from core.utils import load_mnist
from core.optimizer import SGD
from core.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt

np.random.seed(1234)

model = Model()

model.add(Conv2D(filters=1, shape=(28, 28, 1), kernel_size=(3, 3)))
model.add(ReLU())
model.add(MaxPooling2D(shape=(2,2)))
model.add(Flatten())
model.add(Dense(shape=(169, 128))) #676
model.add(ReLU())
model.add(Dense(shape=(128, 10)))
model.add(Softmax())

model.compile(optimizer=SGD(lr=0.01), loss=CategoricalCrossEntropy())

(x, y), (x_test, y_test) = load_mnist()
x = x.reshape(x.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

train_loss_cnn, train_acc_cnn, val_loss_cnn, val_acc_cnn = model.fit(x, y, x_test, y_test, epoch=10, batch_size=32)

plt.plot(train_acc_cnn, label='cnn train accuracy')
plt.plot(val_acc_cnn, label='cnn val accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
