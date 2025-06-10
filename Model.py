from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from keras import layers, models

import Load

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
    
def main():
    x_train, y_train = Load.load_data('data/mnist_train.csv')
    x_test, y_test = Load.load_data('data/mnist_test.csv')
    model = MyModel(num_labels=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, 
                    validation_split=0.1)
    model.evaluate(x_test, y_test)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig("training_plot.png")

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(test_acc)

if __name__ == '__main__':
    main()
