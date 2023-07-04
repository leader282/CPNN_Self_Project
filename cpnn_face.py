import numpy as np
from keras.utils import np_utils
from random import shuffle
from tqdm import tqdm
import cv2
import os
from dense import Dense
from convolutional_cpnn import Convolutional_cpnn
from convolutional_cnn import Convolutional_cnn
from avg_pooling import AvgPool
from reshape import Reshape
from activations import Relu, Softmax, Sigmoid
from losses import mse, mse_prime, binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict
import matplotlib.pyplot as plt

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 64, 64)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = y.reshape(len(y), 10, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, y_train = preprocess_data(x_train, y_train, 100)
# x_test, y_test = preprocess_data(x_test, y_test, 100)

def my_label(image_name):
    name = image_name.split('.')[0] 
    # if you have two person in your dataset
    if name=="KA":
        return np.array([1,0,0,0,0,0,0,0,0,0])
    elif name=="KL":
        return np.array([0,1,0,0,0,0,0,0,0,0])
    elif name=="KM":
        return np.array([0,0,1,0,0,0,0,0,0,0])
    elif name=="KR":
        return np.array([0,0,0,1,0,0,0,0,0,0])
    elif name=="MK":
        return np.array([0,0,0,0,1,0,0,0,0,0])
    elif name=="NA":
        return np.array([0,0,0,0,0,1,0,0,0,0])
    elif name=="NM":
        return np.array([0,0,0,0,0,0,1,0,0,0])
    elif name=="TM":
        return np.array([0,0,0,0,0,0,0,1,0,0])
    elif name=="UY":
        return np.array([0,0,0,0,0,0,0,0,1,0])
    else:
        return np.array([0,0,0,0,0,0,0,0,0,1])

def my_data():
    data = []
    for img in tqdm(os.listdir("jaffedbase")):
        try:
            path=os.path.join("jaffedbase",img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (64,64))
            data.append([np.array(img_data), my_label(img)])
        except:
            pass
    shuffle(data)
    return data

data = my_data()

train_1 = data[:50]
test = data[50:]
x_train = np.array([i[0] for i in train_1]).reshape(-1, 64, 64, 1)
y_train = np.array([i[1][0] for i in train_1])
x_test = np.array([i[0] for i in test]).reshape(-1, 64, 64, 1)
y_test = np.array([i[1][0] for i in test])

x_train, y_train = preprocess_data(x_train, y_train, 50)
x_test, y_test = preprocess_data(x_test, y_test, 150)



# neural network
# cpnn
network1 = [
    Convolutional_cpnn((1, 64, 64), 3, 12),
    Sigmoid(),
    AvgPool((1, 62, 62), 12),
    Convolutional_cpnn((1, 31, 31), 2, 35),
    Sigmoid(),
    AvgPool((1, 30, 30), 35),
    Convolutional_cpnn((1, 15, 15), 2, 70),
    Sigmoid(),
    AvgPool((1, 14, 14), 70),
    Reshape((70, 7, 7), (70 * 7 * 7, 1)),
    Dense(70 * 7 * 7, 100),
    Softmax(),
    Dense(100, 10),
    Softmax()
]

# cnn
network2 = [
    Convolutional_cnn((1, 64, 64), 3, 12),
    Sigmoid(),
    AvgPool((1, 62, 62), 12),
    Convolutional_cnn((1, 31, 31), 2, 35),
    Sigmoid(),
    AvgPool((1, 30, 30), 35),
    Convolutional_cnn((1, 15, 15), 2, 70),
    Sigmoid(),
    AvgPool((1, 14, 14), 70),
    Reshape((70, 7, 7), (70 * 7 * 7, 1)),
    Dense(70 * 7 * 7, 100),
    Softmax(),
    Dense(100, 10),
    Softmax()
]

# train
# cpnn

print("CPNN :")
plot_iter_cpnn, plot_error_cpnn = train(
    network1,
    mse,
    mse_prime,
    x_train,
    y_train,
    epochs=100,
    learning_rate=0.005,
)

# cnn

print("CNN :")
plot_iter_cnn, plot_error_cnn = train(
    network2,
    mse,
    mse_prime,
    x_train,
    y_train,
    epochs=100,
    learning_rate=0.005,
)

plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Learning Curve")
plt.plot(plot_iter_cpnn, plot_error_cpnn, label="CPNN")
plt.plot(plot_iter_cnn, plot_error_cnn, label="CNN")
plt.legend(loc="best", shadow="True",  fontsize="large")
plt.show()

# # test
# # cpnn
# print("CPNN: ")
# for x, y in zip(x_test, y_test):
#     output = predict(network1, x)
#     if np.argmax(output) != np.argmax(y):
#         print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

# # cnn
# print("CNN :")
# for x, y in zip(x_test, y_test):
#     output = predict(network1, x)
#     if np.argmax(output) != np.argmax(y):
#         print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")