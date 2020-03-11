from keras import optimizers, losses, activations, models
from keras import layers
# from keras_contrib.layers import CRF
from utils import train_val_test



def get_base_model():
    model = models.Sequential()

    # model.add(layers.Permute((2, 1), input_shape=(22, 1000)))
    model.add(layers.Conv1D(32, kernel_size=20, strides=4, input_shape=(1000, 22)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=4, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(64, kernel_size=20, strides=4, input_shape=(1000, 22)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='softmax'))

    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    return model

def cnn():
    base = get_base_model()

    model = models.Sequential()
    # model.add(layers.Input(shape=(None, 1000, 22)))
    model.add(layers.TimeDistributed(base))
    model.add(layers.Conv1D(128, kernel_size=3))
    model.add(layers.SpatialDropout1D(rate=0.01))
    model.add(layers.Conv1D(128, kernel_size=3))
    model.add(layers.Dropout(rate=0.05))
    model.add(layers.Conv1D(4, kernel_size=3, activation='softmax'))
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    return model

if __name__ == '__main__':
    Xtrain, ytrain, Xval, yval = train_val_test()
    model = cnn()
    loss_hist = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), epochs=500)
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
