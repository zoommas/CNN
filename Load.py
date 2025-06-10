import numpy as np

IMAGE_SIZE = 28

"""
Loads in MNIST training dataset. Splits dataset into training set
:return: 3D Tensor input of training/validation set and one hot encoded image labels

"""

def load_data(data_path):
    


    training_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    training_data = training_data[~np.isnan(training_data).any(axis=1)]
    x_train = training_data[:, 1:]
    y_train = training_data[:, 0]
    y_train = (np.arange(10) == y_train[:, None]).astype(np.float32)
    x_train = x_train.reshape(len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1)
    x_train = x_train.astype(np.float32) / 255.0

    return x_train, y_train


