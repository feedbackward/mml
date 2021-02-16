'''Utilities: clerical utilities are placed here.'''

## External modules.
import os
import pprint


###############################################################################


def makedir_safe(dirname):
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return None


def para_shape_check(paras, shapes):
    '''
    A clerical function which checks the shapes
    of parameters against a pre-specified benchmark.
    '''

    if paras is None or shapes is None:
        raise ValueError("Both paras and shapes cannot be None.")

    for pn, p in paras.items():

        try:
            if p.shape != shapes[pn]:
                raise ValueError(
                    "Shape of {} is {}; should be {}.".format(
                        pn, p.shape, shapes[pn]
                    )
                )
        except KeyError:
            print("Parameter {} doesn't match any shape names.".format(pn))
    
    return None


###############################################################################
