import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from random import shuffle
from tqdm import tqdm
import cv2
import os

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict
import matplotlib.pyplot as plt

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, y_train = preprocess_data(x_train, y_train, 100)
# x_test, y_test = preprocess_data(x_test, y_test, 100)

def my_label(image_name):
    name = image_name.split('.')[0] 
    # if you have two person in your dataset
    if name=="Aritra":
        return np.array([1,0])
    elif name=="Babui":
        return np.array([0,1])

def my_data():
    data = []
    for img in tqdm(os.listdir("data")):
        try:
            path=os.path.join("data",img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (28,28))
            data.append([np.array(img_data), my_label(img)])
        except:
            pass
    shuffle(data)
    return data

data = my_data()

train_1 = data[:160]
test = data[40:]
x_train = np.array([i[0] for i in train_1]).reshape(-1, 28, 28, 1)
y_train = np.array([i[1][0] for i in train_1])
x_test = np.array([i[0] for i in test]).reshape(-1, 28, 28, 1)
y_test = np.array([i[1][0] for i in test])

x_train, y_train = preprocess_data(x_train, y_train, 80)
x_test, y_test = preprocess_data(x_test, y_test, 40)



# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Convolutional((1, 26, 26), 3, 5),
    Sigmoid(),
    Reshape((5, 24, 24), (5 * 24 * 24, 1)),
    Dense(5 * 24 * 24, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=40,
    learning_rate=0.01
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    if np.argmax(output) != np.argmax(y):
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")




# minimum error mnist
# 20/20, error=0.0025209156074226242
# 20/20, error=0.006513983985691288

# minimum error face image
# 0.104 (with)
# 0.175



# def data_for_visualization():
#     Vdata = []
#     for img in tqdm(os.listdir("img_for_vis")):
#         try:
#             path = os.path.join("img_for_vis", img)
#             img_num = img.split('.')[0] 
#             img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             img_data = cv2.resize(img_data, (28,28))
#             Vdata.append([np.array(img_data), img_num])
#         except:
#             pass
#     shuffle(Vdata)
#     return Vdata

# Vdata = data_for_visualization()

# x_cross = np.array([i[0] for i in Vdata]).reshape(-1, 28, 28, 1)
# y_cross = np.array([i[1][0] for i in Vdata])

# for x, y in zip(x_cross, y_cross):
#     output = predict(network, x)
#     if np.argmax(output) != np.argmax(y):
#         print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")