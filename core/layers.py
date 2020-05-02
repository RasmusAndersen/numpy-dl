import numpy as np
from core.initializer import GlorotNormal, Normal, Zeros


class Layer(object):
    def __init__(self, shape=None, name=None):
        self.name = name
        self.last_input = None
        self.trainable = False
        self.shape = shape

    def get_name(self):
        return self.name

    def forward(self, x):
        raise NotImplementedError

    def backward(self, prev_delta):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, shape, name=None, initializer=Normal(loc=0, scale=0.01)):
        super(Dense, self).__init__(shape=shape, name=name)
        self.initializer = initializer
        self.trainable = True

        self.weights = self.initializer(shape)
        self.bias = Zeros()(self.weights.shape[-1])

        self.weights_grad = None
        self.bias_grad = None

    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, prev_delta):
        # nice explanation: https://youtu.be/tIeHLnjs5U8?t=281
        #activation_gradient = activation.gradient() * prev_delta
        self.weights_grad = np.dot(self.last_input.T, prev_delta)
        self.bias_grad = np.mean(prev_delta, axis=0)
        return np.dot(prev_delta, self.weights.T)


class Conv2D(Layer):
    def __init__(self, filters, shape, kernel_size, stride=1, name=None, initializer=Normal(loc=0, scale=0.01)):
        super(Conv2D, self).__init__(shape=shape, name=name)
        self.initializer = initializer
        self.trainable = True

        assert len(shape) == 3  # (image_height, image_width, num_channels)
        self.input_size = shape

        assert len(kernel_size) == 2  # (kernel_height, kernel_width)
        self.kernel_size = kernel_size

        self.nb_filters = filters
        self.stride = stride

        self.weights = self.initializer((self.nb_filters, self.input_size[-1], self.kernel_size[0], self.kernel_size[1]))
        self.bias = Zeros()(self.nb_filters)

        self.weights_grad = Zeros()(shape=self.weights.shape)
        self.bias_grad = Zeros()(shape=self.bias.shape)

    def forward(self, x):
        self.last_input = x

        assert len(x.shape) == 4  # (batch, channels, image_height, image_width)

        nb_batches = x.shape[0]
        img_height, img_width = x.shape[1:3]

        output_height = (img_height - self.kernel_size[0]) // self.stride + 1
        output_width = (img_width - self.kernel_size[1]) // self.stride + 1

        output = Zeros()(shape=(nb_batches,
                                output_height,
                                output_width,
                                self.nb_filters
                                )
                         )

        kernel_height, kernel_width = self.kernel_size

        for batch in np.arange(nb_batches):
            for filter in np.arange(self.nb_filters):
                for h in np.arange(img_height-kernel_height):
                    for w in np.arange(img_width-kernel_width):
                        patch = x[batch, h:h+kernel_height, w:w+kernel_width, :]
                        output[batch, h, w, filter] = np.sum(patch * self.weights[filter] + self.bias[filter])

        return output

    def backward(self, prev_delta):
        kernel_height, kernel_width = self.kernel_size
        nb_batch, input_img_height, input_img_width, nb_input_channels = self.last_input.shape

        for filter in np.arange(self.nb_filters):
            for input_channel in np.arange(nb_input_channels):
                for h in np.arange(kernel_height):
                    for w in np.arange(kernel_width):
                        patch = self.last_input[:, h:h+input_img_height-kernel_height+1:self.stride, w:w+input_img_width-kernel_width+1:self.stride, input_channel]
                        grad_window = prev_delta[..., filter]
                        self.weights_grad[filter, input_channel, h, w] = np.sum(patch * grad_window) / nb_batch

        for filter in np.arange(self.nb_filters):
            self.bias_grad[filter] = np.sum(prev_delta[..., filter]) / nb_batch

        kernel_grads = Zeros()(shape=self.last_input.shape)
        for batch in np.arange(nb_batch):
            for filter in np.arange(self.nb_filters):
                for input_channel in np.arange(nb_input_channels):
                    for h in np.arange(input_img_height - kernel_height):
                        for w in np.arange(input_img_width - kernel_width):
                            kernel_grads[batch, h:h + kernel_height, w:w + kernel_width, input_channel] += self.weights[filter, input_channel] * prev_delta[batch, h, w, filter]
        return kernel_grads


class Dropout(Layer):
    def __init__(self, prob, name=None):
        assert (prob > 0.) and (prob < 1.)
        self.prob = prob
        super(Dropout, self).__init__(name=name)

    def forward(self, x, training=False):
        mask = np.random.random(x.shape) > self.prob
        self.last_input = mask
        return x * mask

    def backward(self, prev_delta):
        return prev_delta * self.last_input


class Flatten(Layer):
    def __init__(self, name=None):
        super(Flatten, self).__init__(name=name)

    def forward(self, x):
        self.last_input = x.shape
        return x.reshape((x.shape[0], -1))

    def backward(self, prev_delta):
        return prev_delta.reshape(self.last_input)


class ReLU(Layer):
    def __init__(self, name=None):
        super(ReLU, self).__init__(name=name)

    def forward(self, x, gradient=False):
        self.last_input = x
        return np.maximum(0.0, x)

    def backward(self, prev_delta):
        return (self.last_input > 0).astype(np.float64) * prev_delta


class Softmax(Layer):
    def __init__(self, name=None):
        super(Softmax, self).__init__(name=name)

    def forward(self, x, gradient=False):
        self.last_input = x

        m = np.atleast_2d(x.max(axis=1)).T
        return (np.exp(x - m).T / np.sum(np.exp(x - m).T, axis=0)).T

    def backward(self, prev_delta):
        return np.ones_like(self.last_input) * prev_delta
