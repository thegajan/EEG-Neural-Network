import numpy as np


#############################################################
# Load training and test data
#############################################################
def load_data(verbose=False):
    # load data from local data directory
    X_train_valid = np.load("data/X_train_valid.npy")
    y_train_valid = np.load("data/y_train_valid.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    person_train_valid = np.load("data/person_train_valid.npy")
    person_test = np.load("data/person_test.npy")
    #print verbose output
    if verbose:
        print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
        print ('Test data shape: {}'.format(X_test.shape))
        print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
        print ('Test target shape: {}'.format(y_test.shape))
        print ('Person train/valid shape: {}'.format(person_train_valid.shape))
        print ('Person test shape: {}'.format(person_test.shape))

    return X_train_valid, y_train_valid, X_test, y_test, person_train_valid, person_test


#############################################################
# Training, Validation, Test Set
#############################################################
def train_val_test():
    X_train_valid, y_train_valid, X_test, y_test, person_train_valid, person_test = load_data()
    y_train_valid -= 769
    y_test -= 769
    perm = np.random.permutation(X_train_valid.shape[0])
    numTrain = int(0.8*X_train_valid.shape[0])
    numVal = X_train_valid.shape[0] - numTrain
    Xtrain = X_train_valid[perm[0:numTrain]]
    ytrain = y_train_valid[perm[0:numTrain]]
    Xval = X_train_valid[perm[numTrain: ]]
    yval = y_train_valid[perm[numTrain: ]]
    return Xtrain, ytrain, Xval, yval


#############################################################
# Load subject 1 data
#############################################################
def load_subject1():
    # load data
    X_train_valid, y_train_valid, X_test, y_test, person_train_valid, person_test = load_data()

    Xone_train_val = []
    yone_train_val = []
    for i in range(len(person_train_valid)):
        if person_train_valid[i] == 1:
            Xone_train_val.append(X_train_valid[i])
            yone_train_val.append(y_train_valid[i])
    Xone_test = []
    yone_test = []
    for i in range(len(person_test)):
        if person_test[i] == 1:
            Xone_test.append(X_test[i])
            yone_test.append(y_test[i])

    Xone_train_val = np.array(Xone_train_val)
    yone_train_val = np.array(yone_train_val)
    Xone_test = np.array(Xone_test)
    yone_test = np.array(yone_test)

    yone_train_val -= 769
    yone_test -= 769

    perm = np.random.permutation(Xone_train_val.shape[0])
    numTrain = int(0.8*Xone_train_val.shape[0])
    numVal = Xone_train_val.shape[0] - numTrain
    Xonetrain = Xone_train_val[perm[0:numTrain]]
    yonetrain = yone_train_val[perm[0:numTrain]]
    Xoneval = Xone_train_val[perm[numTrain: ]]
    yoneval = yone_train_val[perm[numTrain: ]]

    return Xonetrain, Xoneval, yonetrain, yoneval, Xone_test, yone_test


if __name__ == '__main__':
    load_data(verbose=True)
    Xone_train, Xone_val, yone_train, yone_val, Xone_test, yone_test = load_subject1()
    print(Xone_train.shape)
    print(Xone_val.shape)
    print(yone_train.shape)
    print(yone_val.shape)
    print(Xone_test.shape)
    print(yone_test.shape)
