'''Losses: binary classification margin and related losses.'''

## External modules.
from copy import deepcopy
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class Margin_Binary(Loss):
    '''
    Margin for binary classification.
    X: (n, num_features)
    y: (n, 1), taking values in {-1,+1}.
    '''
    
    def __init__(self, hinge=True, threshold=0.0, name=None):
        self.hinge = hinge
        self.threshold = threshold
        super().__init__(name=name)
        return None
    
    
    def func(self, model, X, y):
        '''
        Assumes the model returns a score for the
        positive class.
        '''
        S = model(X) # raw scores, should have shape (n,1).
        if S.ndim != y.ndim:
            raise ValueError("S and y do not have the same number of axes.")
        elif len(S) != len(y):
            raise ValueError("The number of data in S and y do not match.")
        elif S.shape[1] != y.shape[1]:
            raise ValueError("The shape of S and y do not match.")
        else:
            L = S*y
            if self.hinge:
                return np.where(L >= self.threshold, L, 0.0)
            else:
                return L
    
    
    def grad(self, model, X, y):
        '''
        Assumes that model.grad returns gradients
        with shape (n, num_features, 1), and then
        returns the loss gradient with same shape.
        '''
        
        S = model(X) # raw scores, should have shape (n,1).
        if S.ndim != y.ndim:
            raise ValueError("S and y do not have the same number of axes.")
        elif len(S) != len(y):
            raise ValueError("The number of data in S and y do not match.")
        elif S.shape[1] != y.shape[1]:
            raise ValueError("The shape of S and y do not match.")
        else:

            ## First compute the relevant coefficients.
            if self.hinge:
                coeffs = np.where(S*y >= self.threshold, y, 0.0)
            else:
                coeffs = y
            
            ## Change from (n, 1) to (n, 1, 1) for broadcasting.
            coeffs_exp = np.expand_dims(coeffs, axis=1)
            
            ## Final gradient computations.
            loss_grads = deepcopy(model.grad(X=X))
            for pn, g in loss_grads.items():
                
                ## Before updating, do a shape check to be safe.
                if g.ndim != coeffs_exp.ndim:
                    raise ValueError("g.ndim != coeffs_exp.ndim.")
                elif g.shape[0] != len(coeffs_exp):
                    raise ValueError("g.shape[0] != len(coeffs_exp).")
                elif g.shape[2] != coeffs_exp.shape[2]:
                    raise ValueError("g.shape[2] != coeffs_exp.shape[2].")
                else:
                    g *= coeffs_exp
        
            return loss_grads


###############################################################################
