'''Losses: absolute penalty function.'''

## External modules.
from copy import deepcopy
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class Absolute(Loss):
    '''
    '''
    
    def __init__(self, name=None):
        super().__init__(name=name)
        return None

    
    def func(self, model, X, y):
        '''
        '''
        return np.absolute(model(X=X)-y)


    def grad(self, model, X, y):
        '''
        '''
        loss_grads = deepcopy(model.grad(X=X)) # start with model grads.
        signs = np.sign(model(X=X)-y) # loss sub-gradient (non-composite).

        ## Shape check to be safe.
        if signs.ndim != 2:
            raise ValueError("Require model(X)-y to have shape (n,1).")
        elif signs.shape[1] != 1:
            raise ValueError("Only implemented for single-output models.")
        else:
            for pn, g in loss_grads.items():
                g *= np.expand_dims(a=signs,
                                    axis=tuple(range(2,g.ndim)))
        return loss_grads


###############################################################################
