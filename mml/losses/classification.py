'''Losses: classification error functions.'''

## Internal modules.
from mml.losses import Loss
from mml.utils.linalg import onehot


###############################################################################


class Zero_One(Loss):
    
    def __init__(self, name=None):
        super().__init__(name=name)
        return None

    
    def func(self, model, X, y):
        '''
        This classification error function is based
        upon the following key assumptions:
        - The output of model(X=X) has shape (n, num_classes),
          and the elements in the jth column represent scores
          in favor of the jth class.
        - Labels y are one-hot, i.e., (n, num_classes) shape.
        '''

        num_classes = y.shape[1]

        ## Convert scores into a one-hot prediction.
        y_hat = onehot(y=model(X=X).argmax(axis=1),
                       num_classes=num_classes)

        ## Compare with true one-hot labels.
        return (y_hat != y).any(axis=1, keepdims=True).astype(int)
        

###############################################################################
