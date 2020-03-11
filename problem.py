from keras import layers, models, regularizers
from utils import *
from model_sandbox import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import time

def prob1a(Xonetrain, yonetrain, Xoneval, yoneval, pickle_name):
    CNN2_RNN2(Xonetrain, yonetrain, Xoneval, yoneval, pickle_name)

def prob1b(Xone_test, yone_test):
    #load generally trained model model
    mdl = models.load_model('mdl_wts69.hdf5')
    predictions = mdl.predict(Xone_test)
    y_predict = np.argmax(predictions, axis=1)

    sum = 0
    for i in range(len(y_predict)):
        sum += (y_predict[i] == yone_test[i])
    score = sum / len(y_predict)
    return score

def prob3(X_train, y_train, Xval, yval, pickle_name1, pickle_name2, pickle_name3, pickle_name4):
    time1 = 10
    time2 = 100
    time3 = 500
    CNN2_RNN2(X_train_valid[:,:,:time1], y_train_valid[:,:,:time1], Xval[:,:,:time1], yval[:,:,:time1], pickle_name1)
    CNN2_RNN2(X_train_valid[:,:,:time2], y_train_valid[:,:,:time2], Xval[:,:,:time2], yval[:,:,:time2], pickle_name2)
    CNN2_RNN2(X_train_valid[:,:,:time3], y_train_valid[:,:,:time3], Xval[:,:,:time3], yval[:,:,:time3], pickle_name3)
    CNN2_RNN2(X_train_valid, y_train_valid, Xval, yval, pickle_name4)



if __name__ == '__main__':
    Xtrain, ytrain, Xval, yval = train_val_test()

    # subject one data
    Xonetrain, Xoneval, yonetrain, yoneval, Xone_test, yone_test = load_subject1()

    prob1a(Xonetrain, yonetrain, Xoneval, yoneval, 'prob1a')
    #prob1b(Xone_test, yone_test)
    #prob3(Xtrain, ytrain, Xval, yval, 'time_10', 'time_100', 'time_500', 'time_1k')
