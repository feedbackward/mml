'''Models: base class definitions.'''


###############################################################################


class Model:
    '''
    Model objects represent collections of parametrized
    functions. Each function takes some "inputs" (denoted X),
    and is determined by a dictionary of "parameters" (denoted paras).
    These parameters are the "state" of the Model object, and
    represent a particular choice of candidate from the
    hypothesis class implicitly represented by the Model object.

    Handy references (property, getter/setter):
    https://docs.python.org/3/library/functions.html#property
    https://stackoverflow.com/a/15930977
    '''
    
    def __init__(self, paras_init=None, name=None):
        self._paras = paras_init
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        return None

    
    @property
    def paras(self):
        '''
        Get the current parameter dict.
        '''
        return self._paras

    
    @paras.setter
    def paras(self, paras_new):
        '''
        Set new parameters.
        Can do the entire dictionary all at once,
        or one can do it one element at a time,
        e.g., something like
        >> model.paras["key"] = value
        can be done.
        '''
        self._paras = paras_new
    
    
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
