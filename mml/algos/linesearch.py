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
        
        for pname, p in self.paras.items():
            
            ## Shape check.
            shape_old = p.shape
            shape_new = update_dir[pname].shape
            
            if shape_old != shape_new:
                s_err = "Shapes don't match. {}: {} vs. {}.".format(
                    pname, shape_old, shape_new
                )
                raise RuntimeError(s_err)
            
            ## Assuming shapes match, do additive update.
            p += update_step[pname] * update_dir[pname]

        return None


###############################################################################
