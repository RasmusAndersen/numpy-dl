import numpy as np
from core.activations import ReLU, Softmax
from core.layers import Dense, Dropout
from core.model import Model
from core.utils import load_mnist
from core.optimizer import SGD
from core.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt


layer1 = Dense(shape=(784, 512))
layer2 = Dense(shape=(512, 512))
layer3 = Dense(shape=(512, 10))
activation1 = ReLU()
activation2 = ReLU()
activation3 = Softmax()
dropout1 = Dropout(0.2)
dropout2 = Dropout(0.2)

model = Model()

model.add(layer1)
model.add(activation1)
model.add(dropout1)

model.add(layer2)
model.add(activation2)
model.add(dropout2)

model.add(layer3)
model.add(activation3)

model.compile(optimizer=SGD(lr=0.001), loss=CategoricalCrossEntropy())


(x, y), (x_test, y_test) = load_mnist()

model.fit(x, y, epoch=10)

for i in range(10):
    idx = np.random.choice(np.arange(x_test.shape[0]))
    x_sample = x_test[idx, :].reshape((1, x_test.shape[1]))
    y_sample = y_test[idx, :].reshape((1, y_test.shape[1]))

    prediction = model.predict(x_sample)
    plt.imshow(x_sample.reshape((28, 28)), 'gray')
    plt.title(f'{prediction}\npredicted: {np.argmax(prediction)}, true: {np.argmax(y_sample)}')
    plt.show()
