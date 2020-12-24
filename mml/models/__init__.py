'''Models: base class definitions.'''


###############################################################################


class Model:
    '''
    Model objects represent collections of parametrized
    functions. Each function takes some "inputs" (denoted X),
    and is determined by a "parameter" (denoted w).
    
    The attribute self.w (if not None) is the "state" of
    the Model object, and it represents a particular choice
    of candidate from the hypothesis class implicitly
    represented by the Model object.
    '''
    
    def __init__(self, w_init=None, name=None):
        self.w = w
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        return None
    
    
    def __str__(self):
        out = "Model name: {}".format(self.name)
        return out


    def __call__(self, X=None):
        return self.func(w=self.w, X=X)


    def func(self, w=self.w, X=None):
        '''
        Execute the model on given inputs.
        (implemented in child classes)
        '''
        raise NotImplementedError

    
    def grad(self, w=self.w, X=None):
        '''
        When applicable, compute the gradient with
        respect to parameter w.
        (implemented in child classes)
        '''
        raise NotImplementedError

    
    def hess(self, w=self.w, X=None):
        '''
        When applicable, compute the Hessian with
        respect to parameter w.
        (implemented in child classes)
        '''
        raise NotImplementedError


###############################################################################
