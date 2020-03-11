from keras import layers, models, regularizers
from utils import *
from model_sandbox import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # eeg data
    Xtrain, ytrain, Xval, yval = train_val_test()

    # subject one data
    Xonetrain, Xoneval, yonetrain, yoneval, Xone_test, yone_test = load_subject1()

    #load generally trained model model
    mdl = models.load_model('mdl_wts69.hdf5')
    predictions = mdl.predict(Xone_test)
    y_predict = np.argmax(predictions, axis=1)

    sum = 0
    for i in range(len(y_predict)):
        sum += (y_predict[i] == yone_test[i])
    score = sum / len(y_predict)
    print(score)
