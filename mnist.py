'''
    neural network mnist example
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import os

rand_index = rnd.randint(0, 100)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert pixel values between 0.0-1.0
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def train_and_save_model():
    global x_train, y_train, x_test, y_test

    # model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    # hidden layer ; 128 neurons, rectified linear (activation function)
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

    # output layer ; number of classifications , soft max probability distribution
    # soft max turns numbers into probabilities that sum to one
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # the model is always trying to MINIMIZE loss
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 'sparse_categorical_crossentropy' is used because we aren't limited
    # to how many classes the model can classify

    # lets train the model;
    model.fit(x_train, y_train, epochs=3)

    model.save('trained.model')


def predict_model(model):
    # validation loss/accuracy
    loss, acc = model.evaluate(x_test, y_test)
    print("loss: %s , accuracy: %s" % (loss, acc))

    # calc predictions
    predictions = model.predict([x_test])
    print("prediction: %s" % np.argmax(predictions[rand_index]))

    # show the image
    plt.imshow(x_test[rand_index], cmap='gray')
    plt.show()


if not os.path.isdir(os.curdir + '/trained.model'):  # if the model already exists;
    train_and_save_model()
    predict_model(tf.keras.models.load_model('trained.model'))
else:
    predict_model(tf.keras.models.load_model('trained.model'))
