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

    def backward(self, prev_delta, activation):
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

    def backward(self, prev_delta, activation):
        # nice explanation: https://youtu.be/tIeHLnjs5U8?t=281
        activation_gradient = activation.gradient() * prev_delta
        self.weights_grad = np.dot(self.last_input.T, activation_gradient)
        self.bias_grad = np.mean(activation_gradient, axis=0)
        return np.dot(activation_gradient, self.weights.T)


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

    def backward(self, prev_delta, activation):
        activation_gradient = activation.gradient() * prev_delta

        kernel_height, kernel_width = self.kernel_size
        nb_batch, input_img_height, input_img_width, nb_input_channels = self.last_input.shape

        for filter in np.arange(self.nb_filters):
            for input_channel in np.arange(nb_input_channels):
                for h in np.arange(kernel_height):
                    for w in np.arange(kernel_width):
                        patch = self.last_input[:, h:h+input_img_height-kernel_height+1:self.stride, w:w+input_img_width-kernel_width+1:self.stride, input_channel]
                        grad_window = activation_gradient[..., filter]
                        self.weights_grad[filter, input_channel, h, w] = np.sum(patch * grad_window) / nb_batch

        for filter in np.arange(self.nb_filters):
            self.bias_grad[filter] = np.sum(activation_gradient[..., filter]) / nb_batch

        kernel_grads = Zeros()(shape=self.last_input.shape)
        for batch in np.arange(nb_batch):
            for filter in np.arange(self.nb_filters):
                for input_channel in np.arange(nb_input_channels):
                    for h in np.arange(input_img_height - kernel_height):
                        for w in np.arange(input_img_width - kernel_width):
                            kernel_grads[batch, h:h + kernel_height, w:w + kernel_width, input_channel] += self.weights[filter, input_channel] * activation_gradient[batch, h, w, filter]
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

    def backward(self, prev_delta, activation):
        return prev_delta * self.last_input


class Flatten(Layer):
    def __init__(self, name=None):
        super(Flatten, self).__init__(name=name)
        self.trainable = True

        self.weights, self.bias, self.weights_grad, self.bias_grad = np.arange(4)

    def forward(self, x):
        self.last_input = x.shape
        return x.reshape((x.shape[0], -1))

    def backward(self, prev_delta, activation):
        return prev_delta.reshape(self.last_input)
