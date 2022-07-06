import numpy as np
import skimage.measure
from layer import Layer

class AvgPool(Layer):
    def __init__(self, input_shape, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height//2, input_width//2)
        self.input_height = input_height
        self.input_width = input_width

    def forward(self, input):
        self.input = input
        self.output = np.random.randn(*self.output_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] = skimage.measure.block_reduce(self.input[j], (2,2), np.mean)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input_shape)
        for i in range(min(self.depth, self.input_depth)):
            for j in range(self.input_depth):
                temp = np.concatenate([output_gradient[i], output_gradient[i]])
                temp = np.concatenate([temp, temp])
                input_gradient[j] += 0.25 * temp.reshape(self.input_height, self.input_width)
        return input_gradient
