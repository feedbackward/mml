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


###############################################################################
