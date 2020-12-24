'''Models: linear regression.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models import Model


###############################################################################


class LinearRegression(Model):
    '''
    Linear regression model (one output).
    '''
    
    def __init__(self, w_init, name="Linear regression"):
        super(LinearRegression, self).__init__(w_init=w_init,
                                               name=name)
        return None
    
    
    def func(self, w, X):
        return np.matmul(X,w)
    
    
    def grad(self, w, X):
        return X
    
    
    def hess(self, w, X):
        n, d = X.shape
        return np.zeros((d,d,n), dtype=w.dtype)
    

###############################################################################
