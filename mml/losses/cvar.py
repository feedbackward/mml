'''Losses: logistic loss.'''

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
    '''
    
    def __init__(self, loss_base, quantile, name=None):
        loss_name = "CVaR x {}".format(str(base_loss))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.quantile = quantile
        return None

    
    def func(self, model, X, y, v=0.0):
        '''
        Outputs the "CVaR loss" of the same shape as the
        base loss being used.
        '''
        return v + (1./self.quantile) * np.clip(
            a=self.loss(model=model, X=X, y=y)-v,
            a_min=0.0,
            a_max=None
        )
    
    
    def grad(self, model, X, y, v=0.0):
        '''
        Returns a tuple of sub-gradient arrays.
        One is the array for the sub-gradient taken
        with respect to the main parameter of interest,
        i.e., the "state" of the model.
        The other is "v" here, namely the CVaR shift
        parameter (a scalar).
        '''

        ## Initial computations.
        g_w = self.loss.grad(model=model, X=X, y=y)
        
        l_check = np.clip(a=np.sign(self.loss(model=model, X=X, y=y)-v),
                          a_min=0.0,
                          a_max=None)

        ## Check dimensions.
        ldim = l_check.ndim
        gdim = g_w.ndim

        ## Sub-gradient with respect to main parameters.
        if ldim > gdim:
            raise ValueError("Axis dimensions are wrong; ldim > gdim.")
        elif ldim < gdim:
            l_check_exp = np.expand_dims(
                a=l_check,
                axis=tuple(i for i in range(ldim,gdim))
            )
            g_w *= l_check_exp/self.quantile
        else:
            g_w *= l_check/self.quantile

        ## Sub-gradient with respect to CVaR shift parameter.
        g_v = np.where(l_check>0.0, 1.0-1.0/self.quantile, 1.0)

        ## Return a tuple of the two sub-gradient arrays.
        return (g_w, g_v)

        
###############################################################################
