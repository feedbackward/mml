'''Losses: quadratic penalty function.'''

## Internal modules.
from mml.losses import Loss


###############################################################################


class Quadratic(Loss):
    
    def __init__(self, name=None):
        super(Quadratic, self).__init__(name=name)
        return None

    
    def func(self, model, X, y):
        return (model(X=X)-y)**2 / 2.0


    def grad(self, model, X, y):
        return (model(X=X)-y)*model.grad(X=X)


###############################################################################
