import numpy as np
from core.layers import Dense, Dropout, Conv2D, Flatten, ReLU, Softmax, MaxPooling2D
from core.model import Model
from core.utils import load_mnist
from core.optimizer import SGD
from core.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt

model = Model()

model.add(Conv2D(filters=1, shape=(28, 28, 1), kernel_size=(3, 3)))
model.add(ReLU())
model.add(MaxPooling2D(shape=(2,2)))
model.add(Flatten())
model.add(Dense(shape=(169, 128)))
model.add(ReLU())
model.add(Dense(shape=(128, 10)))
model.add(Softmax())

model.compile(optimizer=SGD(lr=0.01), loss=CategoricalCrossEntropy())

(x, y), (x_test, y_test) = load_mnist()
x = x.reshape(x.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model.fit(x, y, epoch=5, batch_size=32)

for i in range(10):
    idx = np.random.choice(np.arange(x_test.shape[0]))
    x_sample = x_test[idx, :]
    y_sample = y_test[idx, :]

    prediction = model.predict(np.asarray([x_sample]))
    plt.imshow(x_sample[...,0], 'gray')
    plt.title(f'{prediction}\npredicted: {np.argmax(prediction)}, true: {np.argmax(y_sample)}')
    plt.show()