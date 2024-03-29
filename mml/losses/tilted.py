'''Losses: exponential tilting.'''

## External modules.
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class Tilted(Loss):
    '''
    Losses passed through an exponential tilting function.
    - loss_base: the base loss object.
    - tilt: a non-zero value that controls the degree and
      the direction of the "tilt" of the objective.
    '''
    
    def __init__(self, loss_base, tilt, name=None):
        loss_name = "Tilted x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.tilt = tilt
        return None

    
    def base(self, model, X, y):
        '''
        Calls the base loss upon which this
        modified loss is built.
        '''
        return self.loss(model=model, X=X, y=y)
    

    def orig(self, model, X, y):
        '''
        Computes the original Tilted objective,
        in contrast with the modified loss that is
        used in func() and grad(). This involves
        averaging, so this function always returns
        a scalar, not an array.
        '''
        losses = self.loss(model=model, X=X, y=y)
        if self.tilt >= 0.0:
            loss_shift = np.max(losses)
        else:
            loss_shift = np.min(losses)
        return (np.log(
            np.mean(
                np.exp(self.tilt*(losses-loss_shift))
            )
        ) + loss_shift) / self.tilt
    
    
    def func(self, model, X, y):
        '''
        '''
        losses = self.loss(model=model, X=X, y=y)
        if self.tilt >= 0.0:
            loss_shift = np.max(losses)
        else:
            loss_shift = np.min(losses)
        ## NOTE: the loss_shift is to prevent overflow.
        return np.exp(self.tilt*(losses-loss_shift))
    
    
    def grad(self, model, X, y):
        '''
        '''
        tilted_losses = self.func(model=model, X=X, y=y)
        ldim = tilted_losses.ndim
        loss_grads = self.loss.grad(model=model, X=X, y=y)
        
        ## Main gradient computations.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ldim > gdim:
                raise ValueError("Axis dimensions are wrong; ldim > gdim.")
            elif ldim < gdim:
                tilted_losses_exp = np.expand_dims(
                    a=tilted_losses,
                    axis=tuple(range(ldim,gdim))
                )
                g *= tilted_losses_exp * self.tilt
            else:
                g *= tilted_losses * self.tilt
        
        ## Return gradients for all parameters being optimized.
        return loss_grads


###############################################################################
