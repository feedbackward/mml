'''Algorithms: batch gradient descent variants.'''

## External modules.
import numpy as np

## Internal modules.
from mml.algos.linesearch import LineSearch


###############################################################################


class GD_ERM(LineSearch):
    '''
    Empirical risk minimization implemented
    by traditional gradient descent, using a
    fixed step size.
    '''

    def __init__(self, step_coef=None, model=None, loss=None, name=None):
        super(GD_ERM, self).__init__(model=model,
                                     loss=loss,
                                     name=name)
        self.step_coef = step_coef
        return None

    
    def newdir(self, X=None, y=None):
        return -np.mean(self.loss.grad(model=self.model,
                                       X=X, y=y),
                        axis=0, keepdims=True)


    def stepsize(self, newdir=None, X=None, y=None):
        return self.step_coef


###############################################################################
