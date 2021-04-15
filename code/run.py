from preprocess import getData, getDataWithoutNan, splitUpTestingData, splitUpTrainData, mirrorData
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import hyperparameters as hp


def main():
    print("Reached!")
    trainingData, testingData = getData(
        '../data/training.csv', '../data/test.csv')

    train_key_pts, train_imgs = splitUpTrainData(trainingData)
    print("First set of train points", train_key_pts[0])
    print("First set of images", train_imgs[0])

    print("Image dimensions", np.shape(train_imgs))

    print("Number of testing points", train_key_pts.shape[0])
    train_key_pts, train_imgs = getDataWithoutNan(train_key_pts, train_imgs)
    print("Number of testing points without nan", len(train_key_pts))

    train_key_pts, train_imgs = mirrorData(train_key_pts, train_imgs)
    print("Number of testing points with flipping", train_key_pts.shape[0])

    train_imgs = np.reshape(train_imgs, (train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2], 1))

    FPModel = buildModel()
    print(FPModel.summary())

    checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(
        checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    FPModel.fit(train_imgs, train_key_pts, epochs=hp.num_epochs, batch_size=hp.batch_size, callbacks=callbacks_list, validation_split=0.2)


def buildModel():
    FPModel = Sequential()
    FPModel.add(Conv2D(64, 3, activation='relu', input_shape=(hp.image_dim, hp.image_dim, 1), padding='same'))
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


if __name__ == "__main__":
    main()
