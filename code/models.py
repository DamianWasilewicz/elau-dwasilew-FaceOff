import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class FeaturePointModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(FeaturePointModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = hp.learning_rate)

        self.architecture = [
              Conv2D(64,3, activation = 'relu'),
              MaxPool2D(), 
              Conv2D(64,3, activation = 'relu'),
              MaxPool2D(), 
              Conv2D(64,3, activation = 'relu'),
              MaxPool2D(), 
              Conv2D(128,3, activation = 'relu'), 
              MaxPool2D(), 
              Conv2D(128,3, activation = 'relu'), 
              MaxPool2D(), 
              Conv2D(256,3, activation = 'relu'), 
              MaxPool2D(),
              Dropout(rate=0.5), 
              Flatten(), 
              Dense(units=(2 * hp.num_classes), activation='relu'),
              Dense(units=hp.num_classes, activation='softmax')
              ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)


        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.mean_squared_error(labels, predictions)
