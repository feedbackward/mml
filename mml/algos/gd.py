'''Algorithms: batch gradient descent variants.'''

## Internal modules.
from mml.algos.linesearch import LineSearch


###############################################################################


class GD_ERM(LineSearch):
    '''
    Empirical risk minimization implemented
    by traditional gradient descent, using a
    fixed step size for all parameters.
    '''

    def __init__(self, step_coef=None, model=None, loss=None, name=None):
        super().__init__(model=model, loss=loss, name=name)
        self.step_coef = {}
        for pn, p in self.paras.items():
            self.step_coef[pn] = step_coef
        return None

    
    def newdir(self, X=None, y=None):
        loss_grads = self.loss.grad(model=self.model, X=X, y=y)
        newdirs = {}
        for pn, g in loss_grads.items():
            newdirs[pn] = -g.mean(axis=0, keepdims=False)
        return newdirs


    def stepsize(self, newdirs=None, X=None, y=None):
        '''
        Just return the pre-fixed step sizes.
        '''
        return self.step_coef


###############################################################################
