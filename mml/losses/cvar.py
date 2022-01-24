'''Losses: CVaR loss.'''

## External modules.
import numpy as np

## Internal modules.
from mml.losses import Loss


###############################################################################


class CVaR(Loss):
    '''
    A special loss class that takes a base loss
    object upon construction, and uses that to
    make a conditional value at risk (CVaR) type
    of loss.
    - loss_base: the base loss object.
    - alpha: a value between 0 and 1, here this means
      conditioning on exceeding the (1.0-alpha) quantile.
    '''
    
    def __init__(self, loss_base, alpha, name=None):
        loss_name = "CVaR x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.alpha = alpha
        return None


    def base(self, model, X, y):
        '''
        Calls the base loss upon which this
        modified loss is built.
        '''
        return self.loss(model=model, X=X, y=y)

    
    def func(self, model, X, y):
        '''
        '''
        v = model.paras["v"].item()
        return v + (1./self.alpha) * np.clip(
            a=self.loss(model=model, X=X, y=y)-v,
            a_min=0.0,
            a_max=None
        )
    
    
    def grad(self, model, X, y):
        '''
        '''

        ## Initial computations.
        loss_grads = self.loss.grad(model=model, X=X, y=y)
        v = model.paras["v"].item() # extract scalar.
        vdim = model.paras["v"].ndim
        l_check = np.clip(a=np.sign(self.loss(model=model, X=X, y=y)-v),
                          a_min=0.0,
                          a_max=None)
        ldim = l_check.ndim

        ## Main sub-gradient computations.
        ## Note: since "v" isn't part of model grad calcs,
        ##       we never need to worry about "v" getting
        ##       mixed into the loop below.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ldim > gdim:
                raise ValueError("Axis dimensions are wrong; ldim > gdim.")
            elif ldim < gdim:
                l_check_exp = np.expand_dims(
                    a=l_check,
                    axis=tuple(range(ldim,gdim))
                )
                g *= l_check_exp / self.alpha
            else:
                g *= l_check / self.alpha
        
        ## Finally, sub-gradient with respect to CVaR shift parameter.
        loss_grads["v"] = np.expand_dims(
            a=np.where(l_check>0.0, 1.0-1.0/self.alpha, 1.0),
            axis=tuple(range(ldim,1+vdim))
        )
        
        ## Return gradients for all parameters being optimized.
        return loss_grads

        
###############################################################################
