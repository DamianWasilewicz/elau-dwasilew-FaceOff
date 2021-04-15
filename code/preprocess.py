import os
import random
import numpy as np
import pandas as pd
import hyperparameters as hp
from PIL import Image
import tensorflow as tf
from math import isnan


'''
Function that returns DataFrame objects for the training data and for the testing data
'''


def getData(trainpath, testpath):
    train = np.array(pd.read_csv(trainpath))
    test = np.array(pd.read_csv(testpath))
    return train, test

def splitUpTrainData(train):
    train_keypoints = np.array(train[:,:-1]).astype('float32')
    raw_train_imgs = train[:, -1]
    train_imgs = []
    for image in range(raw_train_imgs.shape[0]):
        split_img = np.reshape(np.array(raw_train_imgs[image].split(' ')).astype('float32'), (hp.image_dim, hp.image_dim))
        train_imgs.append(split_img)

    train_imgs = np.array(train_imgs)
    return train_keypoints, train_imgs

def splitUpTestingData(test):
    test_ids = test[:, 0]
    raw_test_imgs = test[:, 1:]
    test_imgs = []
    for image in range(raw_test_imgs): 
        split_img = np.reshape(np.array(raw_test_imgs[image].split(' ')).astype('float32'), (hp.image_dim, hp.image_dim))
        test_imgs.append(split_img)

    test_imgs = np.array(test_imgs)
    return test_ids, np.array(test_imgs)

def getDataWithoutNan(keypts, imgs):
    newKeypts = []
    newImgs = []
    for i in range(keypts.shape[0]):
        if(isComplete(keypts[i])):
            newKeypts.append(keypts[i])
            newImgs.append(imgs[i])
    
    return newKeypts, newImgs


def isComplete(row):
    for i in range(len(row)):
        if isnan(row[i]):
            return False
    return True

def mirrorData(keypts, imgs):
    expanded_pts = keypts
    expanded_imgs = imgs
    for image in range(len(imgs)):
        mirrored_img = np.flip(imgs[image])
        expanded_imgs.append(mirrored_img)
        expanded_pts.append(keypts[image])
    
    return np.array(expanded_pts), np.array(expanded_imgs)
