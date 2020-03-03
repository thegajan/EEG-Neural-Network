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
