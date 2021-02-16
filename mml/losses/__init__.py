'''Losses: base class definitions.'''


###############################################################################


class Loss:
    '''
    Loss objects represent a composition of a
    penalty function and a stateful model, i.e., a
    particular candidate. The final loss is determined
    by up to two additional arguments, namely some
    "inputs" (denoted X) and "outputs" (denoted y).
    
    Any partial derivative calculations are for the
    composition, taken with respect to the parameter that
    determines a model.
    '''
    
    def __init__(self, name=None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        return None
    
    
    def __str__(self):
        '''
        For printing out the relevant loss name.
        '''
        out = "Loss name: {}".format(self.name)
        return out
    
    
    def __call__(self, model=None, X=None, y=None):
        '''
        Lets us compute loss values as loss(model,X,y).
        '''
        return self.func(model=model, X=X, y=y)
    
    
    def func(self, model=None, X=None, y=None):
        '''
        Compute the loss.
        (implemented in child classes)
        '''
        raise NotImplementedError
    
    
    def grad(self, model=None, X=None, y=None):
        '''
        When applicable, compute the loss gradient.
        (implemented in child classes)
        '''
        raise NotImplementedError
    
    
    def hess(self, model=None, X=None, y=None):
        '''
        When applicable, compute the loss Hessian.
        (implemented in child classes)
        '''
        raise NotImplementedError
    
        
###############################################################################
