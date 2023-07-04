# Convolutional Polynomial Neural Network

## Description
This repository contains code for Convolutional Polynomial Neural Network implementation.

The file ```cpnn_1.py``` makes a prediction using Tensorflow's built-in CNN implementation functions and the training and testing is done on real-world data of images of me and my younger brother.

The file ```cpnn.py``` collects samples of data from the camera of the user to be added in the ```img_for_vis``` folder to be later used to predict the results.

The file ```cpnn_face.py``` is the required file with the implementation of the CPNN and compares both the CNN and CPNN loss metric by plotting a comparison using matplotlib. The dataset used is the JAFFE database.

The file ```convolutional_cpnn.py``` is the modified version of the file ```convolutional_cnn.py``` which hosts the major difference in the implementation of the CPNN from the CNN.

The pdf file contains a detailed description of the CPNN model and is the whitepaper.

Other files are helping files that support code snippets for building the whole network.

## Setup
Run ```python -m venv env``` and then install all required libraries in the virtual environment using ```pip install -r requirements.txt```

Then any file can be run as ```python <filename.extension>```
