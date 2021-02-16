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
        super(GD_ERM, self).__init__(model=model,
                                     loss=loss,
                                     name=name)
        self.step_coef = {}
        for pname, p in self.paras.items():
            self.step_coef[pname] = step_coef
        return None

    
    def newdir(self, X=None, y=None):
        return -self.loss.grad(model=self.model,
                               X=X, y=y).mean(axis=0, keepdims=False)


    def stepsize(self, newdir=None, X=None, y=None):
        return self.step_coef


###############################################################################
