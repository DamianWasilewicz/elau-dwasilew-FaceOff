from preprocess import getData, getDataWithoutNan, splitUpTestingData, splitUpTrainData, mirrorData
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
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
    reshaped_test_imgs = np.reshape(test_imgs, (test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2], 1))

    FPModel = buildModel()
    print(FPModel.summary())

    # checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    # checkpoint = ModelCheckpoint(
    #     checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # callbacks_list = [checkpoint]
    loadWeights(FPModel)
    visualize(test_imgs, reshaped_test_imgs, FPModel)

    real_time_prediction(FPModel)







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


def train(train_imgs, train_data, model):
     model.fit(train_imgs, train_key_pts, epochs=hp.num_epochs, batch_size=hp.batch_size, callbacks=callbacks_list, validation_split=0.2)
    
def loadWeights(model):
    weights_file = "Weights-001--3.23472.hdf5"
    model.load_weights(weights_file)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error', 'accuracy'])

def test(testingData, model): 
    numCorrect = 0
    for image in range(testingData.shape[0]):
        img = np.reshape(testingData[image], (1, hp.image_dim, hp.image_dim, 1))
        pred = model.predict(img)
        # if math.abs(pred - )

def visualize(testingData, reshaped_test_imgs, model):
    for image in range(5):
        print("Plotting image")
        plt.imshow(testingData[image])
        reshaped_img = np.reshape(reshaped_test_imgs[image], (1, hp.image_dim, hp.image_dim, 1))
        print("Reshaped img shape", np.shape(reshaped_img))
        predicted_pts = model.predict(reshaped_img)
        print("Predicted pts shape", np.shape(predicted_pts))
        for pt in range(30):
            if (pt % 2 == 0):
                circle=plt.Circle( (predicted_pts[0,pt], predicted_pts[0,pt + 1]), radius=1, color='r')
                plt.gca().add_patch(circle)
        plt.show()
    
def real_time_prediction(model):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        (h,w) = frame.shape[:2]
        frame = np.array(frame, dtype=np.uint8)
        frame = imutils.resize(frame, width=400)
        model_frame = cv2.resize(frame, (hp.image_dim, hp.image_dim))
        gray_image = rgb2gray(model_frame)
        model_input = np.reshape(gray_image, (1, hp.image_dim, hp.image_dim, 1))
        # frame = np.reshape(frame, (1, hp.image_dim, hp.image_dim, 1))
        prd = np.array(model(model_input))
        print("Predictions length", len(prd))
        new_image = frame
        # prd = prd / 96
       
        for pt in range(prd.shape[1]):
            if (pt %2 == 0):
                prd[0][pt] = (prd[0][pt] / 96) * 400
                prd[0][pt + 1] = (prd[0][pt + 1] / 96) * 255
                print("X", prd[0][pt])
                print("Y", prd[0][pt + 1])

    
        
        new_image = cv2.circle(new_image, (int(prd[0][0]), int(prd[0][1])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][2]), int(prd[0][3])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][4]), int(prd[0][5])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][6]), int(prd[0][7])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][8]), int(prd[0][9])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][10]), int(prd[0][11])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][12]), int(prd[0][13])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][14]), int(prd[0][15])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][16]), int(prd[0][17])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][18]), int(prd[0][19])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][20]), int(prd[0][21])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][22]), int(prd[0][23])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][24]), int(prd[0][25])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][26]), int(prd[0][27])), 2, (0,0,255))
        new_image = cv2.circle(new_image, (int(prd[0][28]), int(prd[0][29])), 2, (0,0,255))

        print("Prediction pts shape", np.shape(prd))
        print("Getting image")
        print("Dimensions height", np.shape(frame))
        # circle = cv2.circle(new_image, (48,48), 3, (0, 0, 255))
        cv2.imshow("Frame", new_image)
        key = cv2.waitKey(1) & 0xFF
	    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        fps.update()
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()




if __name__ == "__main__":
    main()
