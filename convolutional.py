import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels_1 = np.random.randn(*self.kernels_shape)
        self.kernels_2 = np.random.randn(*self.kernels_shape)
        self.kernels_3 = np.random.randn(*self.kernels_shape)
        self.kernels_4 = np.random.randn(*self.kernels_shape)
        self.biases_1 = np.random.randn(*self.output_shape)
        self.biases_2 = np.random.randn(*self.output_shape)
        self.biases_3 = np.random.randn(*self.output_shape)
        self.biases_4 = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output_1 = np.copy(self.biases_1)
        self.output_2 = np.copy(self.biases_2)
        self.output_3 = np.copy(self.biases_3)
        self.output_4 = np.copy(self.biases_4)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output_1[i] += signal.correlate2d(self.input[j], self.kernels_1[i, j], "valid")
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output_2[i] += signal.correlate2d(self.input[j] * self.input[j], self.kernels_2[i, j], "valid")
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output_3[i] += signal.correlate2d(self.input[j] * self.input[j] * self.input[j], self.kernels_3[i, j], "valid")
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output_4[i] += signal.correlate2d(self.input[j] * self.input[j] * self.input[j] * self.input[j], self.kernels_4[i, j], "valid")
        self.output = self.output_1 + self.output_2 + self.output_3 + self.output_4
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient_1 = np.zeros(self.kernels_shape)
        kernels_gradient_2 = np.zeros(self.kernels_shape)
        kernels_gradient_3 = np.zeros(self.kernels_shape)
        kernels_gradient_4 = np.zeros(self.kernels_shape)
        input_gradient_1 = np.zeros(self.input_shape)
        input_gradient_2 = np.zeros(self.input_shape)
        input_gradient_3 = np.zeros(self.input_shape)
        input_gradient_4 = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient_1[i, j] += signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient_1[j] += signal.convolve2d(output_gradient[i], self.kernels_1[i, j], "full")
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient_2[i, j] += signal.correlate2d(self.input[j] * self.input[j], output_gradient[i], "valid")
                input_gradient_2[j] += signal.convolve2d(output_gradient[i], self.kernels_2[i, j], "full")
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient_3[i, j] += signal.correlate2d(self.input[j] * self.input[j] * self.input[j], output_gradient[i], "valid")
                input_gradient_3[j] += signal.convolve2d(output_gradient[i], self.kernels_3[i, j], "full")
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient_4[i, j] += signal.correlate2d(self.input[j] * self.input[j] * self.input[j] * self.input[j], output_gradient[i], "valid")
                input_gradient_4[j] += signal.convolve2d(output_gradient[i], self.kernels_4[i, j], "full")

        self.kernels_1 -= learning_rate * kernels_gradient_1
        self.kernels_2 -= learning_rate * kernels_gradient_2
        self.kernels_3 -= learning_rate * kernels_gradient_3
        self.kernels_4 -= learning_rate * kernels_gradient_4
        self.biases_1 -= learning_rate * output_gradient
        self.biases_2 -= learning_rate * output_gradient
        self.biases_3 -= learning_rate * output_gradient
        self.biases_4 -= learning_rate * output_gradient
        input_gradient = input_gradient_1 + input_gradient_2 + input_gradient_3 + input_gradient_4
        return input_gradient
