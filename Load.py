import numpy as np

IMAGE_SIZE = 28

def load_training_data(data_path, validation_size = 500):
    
    training_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    x_train = training_data[:, 1:]
    y_train = training_data[:, 0]
    y_train = (np.arange(10) == y_train[:, None].astype(np.float32))

    x_train, x_val, y_train, y_val = x_train[0:(len(x_train) - validation_size), :], x_train[len(
        x_train) - validation_size:len(x_train), :], \
                                     y_train[0:(len(y_train) - validation_size), :], y_train[len(
        y_train) - validation_size: len(y_train), :]
    
    x_train = x_train.reshape(len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1)
    x_val = x_val.reshape(len(x_val), IMAGE_SIZE, IMAGE_SIZE, 1)

    return x_train, x_val, y_train, y_val




