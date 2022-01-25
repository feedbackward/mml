'''Losses: for implementing DRO using Cressie-Read divergences.'''

## External modules.
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class DRO_CR(Loss):
    '''
    Losses viewed through the lense of DRO using Cressie-Read (CR)
    divergences, thus DRO_CR naming.
    - loss_base: the base loss object.
    - shape: the shape parameter c, when (x**c-cx+c-1)/(c(c-1)) is
      the "f" function used in defining the divergence.
    - bound: upper bound on divergence in the DRO constraint.
    '''
    
    def __init__(self, loss_base, bound, shape, name=None):
        loss_name = "DRO_CR x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.shape = shape
        self.bound = bound
        return None
    
    
    def base(self, model, X, y):
        '''
        Calls the base loss upon which this
        modified loss is built.
        '''
        return self.loss(model=model, X=X, y=y)


    def orig(self, model, X, y):
        '''
        Computes the original DRO_CR objective,
        in contrast with the modified loss that is
        used in func() and grad(). This involves
        averaging, so this function always returns
        a scalar, not an array.
        '''
        theta = model.paras["theta"].item()
        cstar = self.shape / (self.shape-1.0)
        crecip = 1.0 / self.shape
        scale = (1.0+self.shape*(self.shape-1.0)*self.bound)**crecip
        return theta + scale * np.mean(
            clip(
                a=self.loss(model=model, X=X, y=y)-theta,
                a_min=0.0, a_max=None
            )**cstar
        )**(1.0/cstar)
    
    
    def func(self, model, X, y):
        '''
        '''
        cstar = self.shape / (self.shape-1.0)
        theta = model.paras["theta"].item()
        return theta + np.clip(a=self.loss(model=model, X=X, y=y)-theta,
                               a_min=0.0, a_max=None)**cstar
    
    
    def grad(self, model, X, y):
        '''
        '''
        ## Initial computations.
        cstar = self.shape / (self.shape-1.0)
        loss_grads = self.loss.grad(model=model, X=X, y=y)
        theta = model.paras["theta"].item() # extract scalar.
        tdim = model.paras["theta"].ndim
        losses = self.loss(model=model, X=X, y=y)
        l_check = np.where(losses>=theta, 1.0, 0.0)
        l_check *= cstar
        l_check *= np.clip(a=losses-theta, a_min=0.0, a_max=None)**(cstar-1.0)
        ldim = l_check.ndim
        
        ## Main sub-gradient computations.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ldim > gdim:
                raise ValueError("Axis dimensions are wrong; ldim > gdim.")
            elif ldim < gdim:
                l_check_exp = np.expand_dims(
                    a=l_check,
                    axis=tuple(range(ldim,gdim))
                )
                g *= l_check_exp
            else:
                g *= l_check

        ## Finally, sub-gradient with respect to shift parameter.
        loss_grads["theta"] = np.expand_dims(
            a=1.0-l_check,
            axis=tuple(range(ldim,1+tdim))
        )
        
        ## Return gradients for all parameters being optimized.
        return loss_grads


###############################################################################
