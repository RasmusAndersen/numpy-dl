import numpy as np
from core.layers import Dense, Dropout, ReLU, Softmax
from core.model import Model
from core.utils import load_mnist
from core.optimizer import SGD
from core.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt

np.random.seed(1234)

model = Model()

model.add(Dense(shape=(784, 512)))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(shape=(512, 512)))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(shape=(512, 10)))
model.add(Softmax())

model.compile(optimizer=SGD(lr=0.001), loss=CategoricalCrossEntropy())


(x, y), (x_test, y_test) = load_mnist()

train_loss, train_acc, val_loss, val_acc = model.fit(x, y, x_test, y_test, epoch=10, batch_size=128)

plt.plot(train_acc, label='train loss')
plt.plot(val_acc, label='val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

