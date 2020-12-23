'''Configuration file for mml library.'''

## External modules.
import os
from pathlib import Path


###############################################################################


## Directory where all original data is to be stored.
dir_data_toread = str(Path.cwd())

## Directory where all processed data is to be stored.
dir_data_towrite = str(Path.cwd())

## Potential alternative setting:
#   os.path.join(str(Path.home()), "data_master")


###############################################################################
