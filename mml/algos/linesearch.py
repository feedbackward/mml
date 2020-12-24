'''Algorithms: base class for iterative algorithms doing line search.'''

## Internal modules.
from mml.algos import Algorithm


###############################################################################


class LineSearch(Algorithm):
    '''
    Line search algorithms iteratively update
    by determining an update direction and then
    specifying a step size.
    '''

    def __init__(self, model=None, loss=None, name=None):
        super(LineSearch, self).__init__(model=model,
                                         loss=loss,
                                         name=name)
        return None

    
    def newdir(self, X=None, y=None):
        '''
        Computing a new direction.
        (implemented in child classes)
        '''
        raise NotImplementedError


    def stepsize(self, newdir=None, X=None, y=None):
        '''
        Computing a step size, given an
        update direction.
        (implemented in child classes)
        '''
        raise NotImplementedError
        

    def update(self, X=None, y=None):
        update_dir = self.newdir(X=X, y=y)
        update_step = self.stepsize(newdir=update_dir, X=X, y=y)
        self.w = self.w + update_step*update_dir


###############################################################################
