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


if __name__ == '__main__':
    load_data(verbose=True)
