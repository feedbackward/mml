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
    
    
    def func(self, paras=None, X=None):
        if paras is None:
            return np.matmul(X,self.paras["w"])
        else:
            return np.matmul(X,paras["w"])
    
    
    def grad(self, paras=None, X=None):
        model_grads = {}
        model_grads["w"] = X # a view.
        return model_grads

    
    def hess(self, paras=None, X=None):
        n, d = X.shape
        model_hessians = {}
        model_hessians["w"] = np.zeros((n,d,d), dtype=X.dtype)
        return model_hessians


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

    
    def func(self, paras=None, X=None):
        '''
        Multi-valued output; shape (n, num_outputs).
        '''
        w = paras["w"] if paras is not None else self.paras["w"]
        return np.matmul(X,w)
    
    
    def grad(self, paras=None, X=None):
        '''
        Returns the Jacobian; shape (n, num_features, num_outputs).
        '''
        if paras is None:
            num_classes = self.paras["w"].shape[1]
        else:
            num_classes = paras["w"].shape[1]
        model_grads = {}
        model_grads["w"] = np.broadcast_to(
            array=np.expand_dims(X, axis=len(X.shape)),
            shape=X.shape+(num_classes,)
        ) # a view.
        return model_grads
    
    
###############################################################################
