from preprocess import getData, getDataWithoutNan, splitUpTestingData, splitUpTrainData, mirrorData
from real_time_prediction import real_time_prediction
from models import buildModel
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.color import rgb2gray
import numpy as np
import hyperparameters as hp
import math
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


def main():
    print("Reached!")
    trainingData, testingData = getData(
        '../data/training.csv', '../data/test.csv')

    # train_key_pts, train_imgs = splitUpTrainData(trainingData)
    # print("First set of train points", train_key_pts[0])
    # print("First set of images", train_imgs[0])

    test_ids, test_imgs = splitUpTestingData(testingData)

    # print("Image dimensions", np.shape(train_imgs))

    # print("Number of testing points", train_key_pts.shape[0])
    # train_key_pts, train_imgs = getDataWithoutNan(train_key_pts, train_imgs)
    # print("Number of testing points without nan", len(train_key_pts))

    # train_key_pts, train_imgs = mirrorData(train_key_pts, train_imgs)
    # print("Number of testing points with flipping", train_key_pts.shape[0])

    # train_imgs = np.reshape(train_imgs, (train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2], 1))
    reshaped_test_imgs = np.reshape(
        test_imgs, (test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2], 1))

    FPModel = buildModel()
    print(FPModel.summary())

    # checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    # checkpoint = ModelCheckpoint(
    #     checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # callbacks_list = [checkpoint]


    # train(train_imgs, train_key_pts, FPModel, callbacks_list)


    loadWeights(FPModel)
    visualize(test_imgs, reshaped_test_imgs, FPModel)

    real_time_prediction(FPModel)





def train(train_imgs, train_key_pts, model, callbacks_list):
    model.fit(train_imgs, train_key_pts, epochs=hp.num_epochs,
              batch_size=hp.batch_size, callbacks=callbacks_list, validation_split=0.2)


def loadWeights(model):
    weights_file = "Weights-001--3.23472.hdf5"
    model.load_weights(weights_file)
    model.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_error', 'accuracy'])


def test(testingData, model):
    numCorrect = 0
    for image in range(testingData.shape[0]):
        img = np.reshape(testingData[image],
                         (1, hp.image_dim, hp.image_dim, 1))
        pred = model.predict(img)
        # if math.abs(pred - )


def visualize(testingData, reshaped_test_imgs, model):
    for image in range(5):
        print("Plotting image")
        plt.imshow(testingData[image])
        reshaped_img = np.reshape(
            reshaped_test_imgs[image], (1, hp.image_dim, hp.image_dim, 1))
        print("Reshaped img shape", np.shape(reshaped_img))
        predicted_pts = model.predict(reshaped_img)
        print("Predicted pts shape", np.shape(predicted_pts))
        for pt in range(30):
            if (pt % 2 == 0):
                circle = plt.Circle(
                    (predicted_pts[0, pt], predicted_pts[0, pt + 1]), radius=1, color='r')
                plt.gca().add_patch(circle)
        plt.show()




if __name__ == "__main__":
    main()
