import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

import hyperparameters as hp


def buildModel():
    FPModel = Sequential()
    FPModel.add(Conv2D(64, 3, activation='relu', input_shape=(
        hp.image_dim, hp.image_dim, 1), padding='same'))
    FPModel.add(MaxPool2D())
    FPModel.add(Conv2D(64, 3, activation='relu', padding='same'))
    FPModel.add(MaxPool2D())
    FPModel.add(Conv2D(64, 3, activation='relu', padding='same'))
    FPModel.add(MaxPool2D())
    FPModel.add(Conv2D(128, 3, activation='relu', padding='same'))
    FPModel.add(MaxPool2D())
    FPModel.add(Conv2D(128, 3, activation='relu', padding='same'))
    FPModel.add(MaxPool2D())
    FPModel.add(Conv2D(256, 3, activation='relu', padding='same'))
    FPModel.add(MaxPool2D())
    FPModel.add(Dropout(rate=0.5))
    FPModel.add(Flatten())
    FPModel.add(Dense(units=(2 * hp.num_classes),
                activation='relu', kernel_initializer='normal'))
    FPModel.add(Dense(units=(2 * hp.num_classes),
                activation='relu', kernel_initializer='normal'))
    FPModel.add(Dense(units=hp.num_classes, activation='linear'))
    FPModel.compile(loss='mean_absolute_error', optimizer='adam',
                    metrics=['mean_absolute_error', 'accuracy'])
    return FPModel



  





