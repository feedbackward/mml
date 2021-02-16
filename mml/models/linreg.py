'''Models: linear regression.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models import Model, init_range
from mml.utils import para_shape_check


###############################################################################


class LinearRegression(Model):
    '''
    Linear regression model, with *one* output.
    Assumes that w_init shape (num_features, 1).
    '''
    
    def __init__(self, num_features,
                 paras_init=None, rg=None,
                 name="Linear regression"):
        
        ## Shape specification.
        self.shapes = {"w": (num_features, 1)}
        
        if paras_init is not None:
            ## If passed initial values, run a shape check.
            para_shape_check(paras=paras_init, shapes=self.shapes)
        else:
            ## If no initial value provided, generate one.
            paras_init = {}
            paras_init["w"] = rg.uniform(low=-init_range,
                                         high=init_range,
                                         size=self.shapes["w"])
        
        ## Main construction.
        super().__init__(paras_init=paras_init, name=name)
        return None
    
    
    def func(self, w=None, X=None):
        if w is None:
            return np.matmul(X,self.paras["w"])
        else:
            return np.matmul(X,w)
    
    
    def grad(self, w=None, X=None):
        return X
    
    
    def hess(self, w=None, X=None):
        n, d = X.shape
        return np.zeros((d,d,n), dtype=self.paras["w"].dtype)


class LinearRegression_Multi(Model):
    '''
    Linear regression model, with *multiple* outputs.
    Assumes that w_init shape (num_features, num_outputs).
    '''
    
    def __init__(self, num_features, num_outputs,
                 paras_init=None, rg=None,
                 name="Multi-output linear regression"):
        
        ## Shape specification.
        self.shapes = {"w": (num_features, num_outputs)}

        if paras_init is not None:
            ## If passed initial values, run a shape check.
            para_shape_check(paras=paras_init, shapes=self.shapes)
        else:
            ## If no initial value provided, generate one.
            paras_init = {}
            paras_init["w"] = rg.uniform(low=-init_range,
                                         high=init_range,
                                         size=self.shapes["w"])

        ## Main construction.
        super().__init__(paras_init=paras_init, name=name)
        return None

    
    def func(self, w=None, X=None):
        '''
        Multi-valued output; shape (n, num_outputs).
        '''
        if w is None:
            return np.matmul(X,self.paras["w"])
        else:
            return np.matmul(X,w)
    
    
    def grad(self, w=None, X=None):
        '''
        Returns the Jacobian; shape (n, num_features, num_outputs).
        '''
        num_classes = self.paras["w"].shape[1] if w is None else w.shape[1]
        return np.broadcast_to(
            array=np.expand_dims(X, axis=len(X.shape)),
            shape=X.shape+(num_classes,)
        )
    
    
###############################################################################
