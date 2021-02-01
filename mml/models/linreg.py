'''Models: linear regression.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models import Model, init_range


###############################################################################


class LinearRegression(Model):
    '''
    Linear regression model, with *one* output.
    Assumes that w_init shape (num_features, 1).
    '''
    
    def __init__(self, num_features,
                 w_init=None, rg=None,
                 name="Linear regression"):
        
        ## Shape specification.
        self.shape = (num_features, 1)

        if w_init is not None:

            ## Shape check.
            if w_init.shape == self.shape:
                _w_init = w_init
            else:
                raise ValueError("w_init has the wrong size.")

        else:
            
            ## If no initial value provided, generate one.
            _w_init = rg.uniform(low=-init_range,
                                 high=init_range,
                                 size=self.shape)
            
        super(LinearRegression, self).__init__(w_init=_w_init,
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
    Linear regression model, with *multiple* outputs.
    Assumes that w_init shape (num_features, num_outputs).
    '''
    
    def __init__(self, num_features, num_outputs,
                 w_init=None, rg=None,
                 name="Multi-output linear regression"):
        
        ## Shape specification.
        self.shape = (num_features, num_outputs)

        if w_init is not None:
            
            ## Shape check.
            if w_init.shape == self.shape:
                _w_init = w_init
            else:
                raise ValueError("w_init has the wrong size.")

        else:
            
            ## If no initial value provided, generate one.
            _w_init = rg.uniform(low=-init_range,
                                 high=init_range,
                                 size=self.shape)
            
        super(LinearRegression_Multi, self).__init__(w_init=_w_init,
                                                     name=name)
        return None

    
    def func(self, w=None, X=None):
        '''
        Multi-valued output; shape (n, num_outputs).
        '''
        if w is None:
            return np.matmul(X,self._w)
        else:
            return np.matmul(X,w)
    
    
    def grad(self, w=None, X=None):
        '''
        Returns the Jacobian; shape (n, num_features, num_outputs).
        '''
        num_classes = self._w.shape[1] if w is None else w.shape[1]
        return np.broadcast_to(
            array=np.expand_dims(X, axis=len(X.shape)),
            shape=X.shape+(num_classes,)
        )
    
    
###############################################################################
