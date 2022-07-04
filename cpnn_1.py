import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import cv2

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

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
            img_data = cv2.resize(img_data, (50,50))
            data.append([np.array(img_data), my_label(img)])
        except:
            pass
    shuffle(data)  
    return data

data = my_data()

train = data[:1600]
test = data[1600:]
X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50, 1)
Y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
Y_test = [i[1] for i in test]

print(Y_train)

tf.compat.v1.reset_default_graph()
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
# 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.fit(X_train, Y_train, n_epoch=12, validation_set=(X_test, Y_test), show_metric = True, run_id="FRS" )


def data_for_visualization():
    Vdata = []
    for img in tqdm(os.listdir("img_for_vis")):
        try:
            path = os.path.join("img_for_vis", img)
            img_num = img.split('.')[0] 
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (50,50))
            Vdata.append([np.array(img_data), img_num])
        except:
            pass
    shuffle(Vdata)
    return Vdata

Vdata = data_for_visualization()

fig = plt.figure(figsize=(20,20))
for num, data in enumerate(Vdata[:20]):
    img_data = data[0]
    y = fig.add_subplot(5,5, num+1)
    image = img_data
    data = img_data.reshape(50,50,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0:
        my_label = 'Aritra'
    else:
        my_label = 'Babui'
        
    y.imshow(image, cmap='gray')
    plt.title(my_label)
    
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
