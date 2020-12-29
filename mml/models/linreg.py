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
    
    
    def func(self, w=None, X=None):
        return np.matmul(X,w)
    
    
    def grad(self, w=None, X=None):
        return X
    
    
    def hess(self, w=None, X=None):
        n, d = X.shape
        return np.zeros((d,d,n), dtype=w.dtype)
    

###############################################################################
