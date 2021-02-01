'''Models: base class definitions.'''


###############################################################################


class Model:
    '''
    Model objects represent collections of parametrized
    functions. Each function takes some "inputs" (denoted X),
    and is determined by a "parameter" (denoted w).
    This parameter is the "state" of the Model object, and
    it represents a particular choice of candidate from the
    hypothesis class implicitly represented by the Model object.

    Handy reference for getters/setters:
    https://stackoverflow.com/a/15930977
    '''
    
    def __init__(self, w_init=None, name=None):
        self._w = w_init
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        return None

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w_new):
        self._w = w_new
    
    
    def __str__(self):
        out = "Model name: {}".format(self.name)
        return out


    def __call__(self, X=None):
        return self.func(w=self._w, X=X)


    def func(self, w=None, X=None):
        '''
        Execute the model on given inputs.
        (implemented in child classes)
        '''
        raise NotImplementedError

    
    def grad(self, w=None, X=None):
        '''
        When applicable, compute the gradient with
        respect to parameter w.
        (implemented in child classes)
        '''
        raise NotImplementedError

    
    def hess(self, w=None, X=None):
        '''
        When applicable, compute the Hessian with
        respect to parameter w.
        (implemented in child classes)
        '''
        raise NotImplementedError


## Range for default (random) parameter initialization.
init_range = 0.05


###############################################################################
