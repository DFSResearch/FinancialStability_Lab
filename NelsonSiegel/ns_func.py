import numpy as np

def F(m, b):
    '''
    Forward rate function: F(m, b)
    Parameters
    ------------
    m: array-like
        maturity vector
    b: array-like
        vector of Beta parameters
    Returns
    ------------
    forward rates vector
    '''
    return b[0] + b[1] * np.exp(-m / b[3]) + b[2] * (m / b[3]) * np.exp(-m / b[3])

def Z(m, b):
    '''
    Spot rate function: Z(m, b)
    Parameters
    ------------
    m: array-like
        maturity vector
    b: array-like
        vector of Beta parameters
    Returns
    ------------
    Spot rates vector
    '''
    return b[0] + (b[1] + b[2]) * (b[3] / m) *(1 - np.exp(-m / b[3])) - b[2] * np.exp(-m / b[3])

def D(m, b):
    '''
    Discount function: D(m, b)
    Parameters
    ------------
    m: array-like
        maturity vector
    b: array-like
        vector of Beta parameters
    Returns
    ------------
    discount factors vector
    '''
    return np.exp(-m * (b[0] + (b[1] + b[2]) * (b[3] / m) *
                        (1 - np.exp(-m / b[3])) - b[2] * np.exp(-m / b[3])))