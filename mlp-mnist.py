import numpy as np
from core.layers import Dense, Dropout, ReLU, Softmax
from core.model import Model
from core.utils import load_mnist
from core.optimizer import SGD
from core.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt


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

model.fit(x, y, epoch=10, batch_size=128)

for i in range(10):
    idx = np.random.choice(np.arange(x_test.shape[0]))
    x_sample = x_test[idx, :].reshape((1, x_test.shape[1]))
    y_sample = y_test[idx, :].reshape((1, y_test.shape[1]))

    prediction = model.predict(x_sample)
    plt.imshow(x_sample.reshape((28, 28)), 'gray')
    plt.title(f'{prediction}\npredicted: {np.argmax(prediction)}, true: {np.argmax(y_sample)}')
    plt.show()
