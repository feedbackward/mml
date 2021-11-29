'''Losses: logistic loss, typical variants.'''

## External modules.
from copy import deepcopy
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class Logistic_Binary(Loss):
    '''
    Classical binary logistic loss.
    X: (n, num_features)
    y: (n, 1), taking values in {-1,+1}.
    '''
    
    def __init__(self, name=None):
        super().__init__(name=name)
        return None
    
    
    def func(self, model, X, y):
        '''
        Assumes model returns a vector of
        *unnormalized* scores for the first class.
        '''

        A = model(X) # raw activations (n, 1).

        if A.ndim != y.ndim:
            raise ValueError("A and y do not have the same number of axes.")
        elif len(A) != len(y):
            raise ValueError("The number of data in A and y do not match.")
        elif A.shape[1] != y.shape[1]:
            raise ValueError("The shape of A and y do not match.")
        else:
            
            ## Get signed activations (noting y is +1/-1 with (n,1) shape).
            A *= y
            
            ## Compute losses while avoiding overflow.
            return np.where(A >= 0.0,
                            np.log1p(np.exp(-A)),
                            np.log1p(np.exp(A))-A)
    
    
    def grad(self, model, X, y):
        '''
        Assumes that model.grad returns a gradient
        with shape (n, num_features, 1), and then
        returns the loss gradient with same shape.
        '''
        
        A = model(X) # raw activations (n, 1).

        if A.ndim != y.ndim:
            raise ValueError("A and y do not have the same number of axes.")
        elif len(A) != len(y):
            raise ValueError("The number of data in A and y do not match.")
        elif A.shape[1] != y.shape[1]:
            raise ValueError("The shape of A and y do not match.")
        else:
            
            ## Get signed activations (noting y is +1/-1 with (n,1) shape).
            A *= y
            
            coeffs = -y * np.where(A >= 0.0,
                                   np.exp(-A)/(1.0+np.exp(-A)),
                                   1.0/(1.0+np.exp(A)))

            ## Change from (n, 1) to (n, 1, 1) for broadcasting.
            coeffs_exp = np.expand_dims(coeffs, axis=1)
            
            ## Final computations.
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
        loss += np.log(np.exp(A_raw-maxes).sum(axis=1,keepdims=True))+maxes
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
        loss_grads = deepcopy(model.grad(X=X))
        for pn, g in loss_grads.items():

            ## Before updating, do a shape check to be safe.
            if g.ndim != D_exp.ndim:
                raise ValueError("g.ndim != D_exp.ndim.")
            elif g.shape[0] != len(D_exp):
                raise ValueError("g.shape[0] != len(D_exp).")
            elif g.shape[2] != D_exp.shape[2]:
                raise ValueError("g.shape[2] != D_exp.shape[2].")
            else:
                g *= D_exp

        return loss_grads
    

###############################################################################
