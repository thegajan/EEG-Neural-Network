from keras import layers, models, regularizers
from utils import train_val_test
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import time


def CNN2_RNN2(Xtrain, ytrain, Xval, yval, pickle_name):
    model = models.Sequential()

    model.add(layers.Permute((2, 1), input_shape=(22, 1000)))
    model.add(layers.Conv1D(40, kernel_size=20, strides=4, input_shape=(1000, 22)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=4, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(40, kernel_size=20, strides=4, input_shape=(1000, 22)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=4, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.LSTM(20, return_sequences=True, stateful=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.LSTM(20, return_sequences=True, stateful=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.8))

    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))

    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

    mcp_save = ModelCheckpoint(pickle_name+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    loss_hist = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=1250, callbacks=[mcp_save])
    model.summary()
    hist = loss_hist.history
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])

    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.show()


def RNN2(X_train_valid, y_train_valid, Xval, yval):
    model = models.Sequential()

    model.add(layers.Permute((2, 1), input_shape=(22, 1000)))
    model.add(layers.LSTM(64, return_sequences=True, stateful=False))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(64, return_sequences=True, stateful=False))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))

    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    # model.summary()
    loss_hist = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=25)
    hist = loss_hist.history
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])

    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.show()


def CNN2_FC(X_train_valid, y_train_valid, Xval, yval):
    model = models.Sequential()

    model.add(layers.Reshape((22, 1000)))
    model.add(layers.Permute((2, 1), input_shape=(22, 1000)))

    model.add(layers.Conv1D(40, kernel_size=20, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(40, kernel_size=20, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=3))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(40, kernel_size=10, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(40, kernel_size=3, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=1))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4))
    model.add(layers.Activation('softmax'))

    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    loss_hist = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=250)
    model.summary()
    hist = loss_hist.history
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])

    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.show()


if __name__ == '__main__':
    Xtrain, ytrain, Xval, yval = train_val_test()
    Xonetrain, Xoneval, yonetrain, yoneval, Xone_test, yone_test = load_subject1()
    # RNN2(Xtrain, ytrain, Xval, yval)
    # CNN2_RNN2(Xtrain, ytrain, Xval, yval)
    # CNN2_FC(Xtrain, ytrain, Xval, yval)

    #CNN2_RNN2(Xonetrain, yonetrain, Xoneval, yoneval)
