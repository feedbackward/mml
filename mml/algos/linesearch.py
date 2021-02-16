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
        super().__init__(model=model, loss=loss, name=name)
        return None

    
    def newdir(self, X=None, y=None):
        '''
        Computing a new direction.
        (implemented in child classes)
        '''
        raise NotImplementedError


    def stepsize(self, newdirs=None, X=None, y=None):
        '''
        Computing a step size, given an
        update direction.
        (implemented in child classes)
        '''
        raise NotImplementedError
        

    def update(self, X=None, y=None):

        newdirs = self.newdir(X=X, y=y)
        update_step = self.stepsize(newdirs=newdirs, X=X, y=y)
        
        for pn, p in self.paras.items():
            
            ## Shape check.
            shape_old = p.shape
            shape_new = newdirs[pn].shape
            
            if shape_old != shape_new:
                s_err = "Shapes don't match. {}: {} vs. {}.".format(
                    pn, shape_old, shape_new
                )
                raise RuntimeError(s_err)
            
            ## Assuming shapes match, do additive update.
            p += update_step[pn] * newdirs[pn]

        return None


###############################################################################
