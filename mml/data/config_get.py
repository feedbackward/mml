'''A simple helper which picks up the master directory for saving data.'''

## External modules.
import os
import sys

## Internal modules.
from mml.config import dir_data_toread
from mml.utils import makedir_safe


###############################################################################


if __name__ == "__main__":

    try:
        newdir = os.path.join(dir_data_toread, sys.argv[1])
        makedir_safe(newdir)
        print(newdir)
    except IndexError:
        print("Please pass some value to this script")


###############################################################################

