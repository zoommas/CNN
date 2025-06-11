import tensorflow as tf
from keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, num_labels=10):
        super(MyModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')
        self.pool1 = layers.MaxPool2D((2, 2))

        self.conv2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')
        self.pool2 = layers.MaxPool2D((2, 2))

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, 'relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_labels)

    def call(self, input, training=False):
        input = self.conv1(input)
        input = self.pool1(input)
        input = self.conv2(input)
        input = self.pool2(input)
        input = self.flatten(input)
        input = self.fc1(input)
        if training:
            input = self.dropout(input)
        input = self.fc2(input)
        return input
    
