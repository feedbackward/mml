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
        if w is None:
            return np.matmul(X,self._w)
        else:
            return np.matmul(X,w)
    
    
    def grad(self, w=None, X=None):
        return X
    
    
    def hess(self, w=None, X=None):
        n, d = X.shape
        return np.zeros((d,d,n), dtype=w.dtype)


class LinearRegression_Multi(Model):
    '''
    Linear regression model (multiple outputs).
    '''
    
    def __init__(self, w_init, name="Multi-output linear regression"):
        super(LinearRegression_Multi, self).__init__(w_init=w_init,
                                                     name=name)
        return None
    
    
    def func(self, w=None, X=None):
        if w is None:
            return np.matmul(X,self._w)
        else:
            return np.matmul(X,w)
    
    
    def grad(self, w=None, X=None):
        '''
        Since this is multi-output, the Jacobian.
        '''
        num_classes = self._w.shape[1] if w is None else w.shape[1]
        return np.broadcast_to(
            array=np.expand_dims(X, axis=len(X.shape)),
            shape=X.shape+(num_classes,)
        )
    
    
###############################################################################
