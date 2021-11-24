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

    if not isinstance(paras, dict):
        raise TypeError("Parameters must be stored in a dict.")

    for pn in shapes.keys():
        
        try:
            p = paras[pn]
            if p.shape != shapes[pn]:
                raise ValueError(
                    "Shape of {} is {}; should be {}.".format(
                        pn, p.shape, shapes[pn]
                    )
                )
            else:
                continue
        
        except KeyError:
            print("Parameter {} wasn't provided.".format(pn))
    
    return None


###############################################################################
