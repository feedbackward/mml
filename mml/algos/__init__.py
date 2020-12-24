'''Algorithms: base class definitions.'''


###############################################################################


class Algorithm:
    '''
    Algorithm objects are iterators which
    interact with both Model and Loss objects
    to construct feedback based on data, and
    based on this feedback they update the state
    of the relevant Model.
    '''

    def __init__(self, model=None, loss=None, name=None):
        self.model = model
        self.w = model.w
        self.loss = loss
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.stop = False
        return None


    def __str__(self):
        out = "Algorithm name: {}".format(self.name)
        return out


    def __iter__(self):
        self.counter = 0
        return self


    def __next__(self):
        if self.stop:
            raise StopIteration
        else:
            return None
        

    def update(self, X=None, y=None):
        '''
        Main parameter update function.
        This can be used by both iterative
        and single-step algorithms.
        (implemented in child classes)
        '''
        raise NotImplementedError


    def check(self):
        '''
        A generic function that checks to see
        if the algorithm should stop.
        '''
        raise NotImplementedError


###############################################################################
