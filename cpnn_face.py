import numpy as np
from keras.utils import np_utils
from random import shuffle
from tqdm import tqdm
import cv2
import os
from dense import Dense
from convolutional import Convolutional
from avg_pooling import AvgPool
from reshape import Reshape
from activations import Sigmoid
from losses import mse, mse_prime
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
network = [
    Convolutional((1, 64, 64), 3, 12),
    Sigmoid(),
    AvgPool((1, 62, 62), 12),
    Convolutional((1, 31, 31), 2, 35),
    Sigmoid(),
    AvgPool((1, 30, 30), 35),
    Convolutional((1, 15, 15), 2, 70),
    Sigmoid(),
    AvgPool((1, 14, 14), 70),
    Reshape((70, 7, 7), (70 * 7 * 7, 1)),
    Dense(70 * 7 * 7, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

# train
plot_iter, plot_error = train(
    network,
    mse,
    mse,
    x_train,
    y_train,
    epochs=100,
    learning_rate=0.01
)

plt.plot(plot_iter, plot_error)
plt.show()

# test
# for x, y in zip(x_test, y_test):
#     output = predict(network, x)
#     if np.argmax(output) != np.argmax(y):
#         print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
