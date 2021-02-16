'''Losses: quadratic penalty function.'''

## External modules.
from copy import deepcopy

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
        loss_grads = deepcopy(model.grad(X=X))
        for pn, g in loss_grads.items():
            g *= model(X=X)-y
        return loss_grads


###############################################################################
