'''Algorithms: robust gradient descent variants.'''

## External modules.
import numpy as np

## Internal modules.
from mml.algos.linesearch import LineSearch


###############################################################################


class RGD_Mest(LineSearch):
    '''
    Robust gradient descent using coordinate-wise M-estimates.

    Reference:
    Efficient learning with robust gradient descent.
    Matthew J. Holland and Kazushi Ikeda.
    Machine Learning, 108(8):1523-1560, 2019.
    '''

    def __init__(self, est_loc, est_scale, delta,
                 mest_thres=1e-03, mest_iters=50,
                 step_coef=None, model=None, loss=None, name=None):
        super().__init__(model=model, loss=loss, name=name)
        self.est_loc = est_loc
        self.est_scale = est_scale
        self.delta = delta
        self.mest_thres = mest_thres
        self.mest_iters = mest_iters
        self.step_coef = {}
        for pn, p in self.paras.items():
            self.step_coef[pn] = step_coef
        return None
    
    
    def newdir(self, X=None, y=None):
        loss_grads = self.loss.grad(model=self.model, X=X, y=y)
        newdirs = {}
        for pn, g in loss_grads.items():

            ## Scale factor (Catoni 2012 style) before std dev estimate.
            s_est = np.sqrt(len(g)/np.log(1.0/self.delta))

            ## Multiply by std dev estimate.
            s_est *= self.est_scale(
                X=g-g.mean(axis=0, keepdims=True)
            )

            ## Location estimate using scaling, negative direction.
            newdirs[pn] = -self.est_loc(X=g, s=s_est,
                                        thres=self.mest_thres,
                                        iters=self.mest_iters)

            ## Ensure shapes match before proceeding.
            newdir_dim = newdirs[pn].ndim
            para_dim = self.paras[pn].ndim
            err_string = "newdirs[pn].shape {}, paras[pn].shape {}".format(
                newdirs[pn].shape, self.paras[pn].shape
            )
            if newdir_dim > para_dim:
                if len(newdirs[pn]) == 1:
                    newdirs[pn] = newdirs[pn][-1]
                else:
                    raise RuntimeError(err_string)
            elif newdir_dim < para_dim:
                raise RuntimeError(err_string)
            else:
                continue
            
        return newdirs
    
    
    def stepsize(self, newdirs=None, X=None, y=None):
        '''
        Just return the pre-fixed step sizes.
        '''
        return self.step_coef


###############################################################################
