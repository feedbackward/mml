'''Losses: logistic loss.'''

## External modules.
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class Logistic(Loss):
    '''
    Assumes the following shapes.
    X: (n, num_features)
    y: (n, num_classes)
    '''
    
    def __init__(self, name=None):
        super().__init__(name=name)
        return None

    
    def func(self, model, X, y):
        '''
        Assumes model returns a vector of
        *unnormalized* scores; one per class.
        '''
        A_raw = model(X) # raw activations (n, num_classes).
        
        ## Initial loss term.
        loss = -np.multiply(A_raw,y).sum(axis=1,keepdims=True)
        
        ## Further computations.
        maxes = A_raw.max(axis=1,keepdims=True) # use to avoid overflow.
        loss += np.log(np.exp(A_raw-maxes).sum(axis=1,keepdims=True)+maxes)
        return loss


    def grad(self, model, X, y):
        '''
        Assumes that model.grad returns a grad/Jacobian
        with shape (n, num_features, num_classes).
        Returns the loss grad/Jacobian with same shape.
        '''

        ## Activations and difference computations.
        D = model(X) # raw activations (n, num_classes).
        D = np.exp(D-D.max(axis=1,keepdims=True)) # avoiding overflow.
        D = np.divide(D,D.sum(axis=1,keepdims=True)) # probabilities.
        D -= y # differences (thus, "D").

        ## Change from (n, num_classes) to (n, 1, num_classes).
        D_exp = np.expand_dims(D, axis=1) # enables broadcasting.

        ## Final computations.
        loss_grads = model.grad(X=X)
        for pn, g in loss_grads.items():
            g *= D_exp
        return loss_grads


###############################################################################
