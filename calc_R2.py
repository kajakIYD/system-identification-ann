import numpy as np

def rSquare(estimations, measurements):
    """ Compute the coefficient of determination of random data.
    This metric gives the level of confidence about the model used to model data"""
    SEE = ((np.array(measurements) - np.array(estimations))**2).sum()
    mMean = (np.array(measurements)).sum() / float(len(measurements))
    dErr = ((mMean - measurements)).sum()

    return 1 - (SEE / dErr)