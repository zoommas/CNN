import tensorflow as tf

from matplotlib import pyplot as plt
from utils.Load import load_data
from model.Model import MyModel

def main():
    x_train, y_train = load_data('data/mnist_train.csv')
    x_test, y_test = load_data('data/mnist_test.csv')
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
    plt.savefig("plots/training_plot.png")

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(test_acc)

if __name__ == '__main__':
    main()
