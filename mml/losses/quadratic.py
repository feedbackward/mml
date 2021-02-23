'''Losses: quadratic penalty function.'''

## External modules.
from copy import deepcopy
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class Quadratic(Loss):
    '''
    '''
    
    def __init__(self, name=None):
        super().__init__(name=name)
        return None

    
    def func(self, model, X, y):
        '''
        '''
        return (model(X=X)-y)**2 / 2.0


    def grad(self, model, X, y):
        '''
        '''
        loss_grads = deepcopy(model.grad(X=X)) # start with model grads.
        diffs = model(X=X)-y # then loss grads (non-composite).

        ## Shape check to be safe.
        if diffs.ndim != 2:
            raise ValueError("Require model(X)-y to have shape (n,1).")
        elif diffs.shape[1] != 1:
            raise ValueError("Only implemented for single-output models.")
        else:
            for pn, g in loss_grads.items():
                g *= np.expand_dims(a=diffs,
                                    axis=tuple(range(2,g.ndim)))
        return loss_grads


###############################################################################
