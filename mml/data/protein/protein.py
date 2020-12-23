'''H5 data prep'''

## External modules.
import csv
import numpy as np
import os
import tables

## Internal modules.
from mml.config import dir_data_toread
from mml.config import dir_data_towrite
from mml.utils import makedir_safe


###############################################################################


## Clerical setup.

data_name = "protein"

toread = os.path.join(dir_data_toread, data_name, "bio_train.dat")
newdir = os.path.join(dir_data_towrite, data_name)
makedir_safe(newdir)
towrite = os.path.join(newdir, "protein.h5")

n_all = 145751
num_features = 74
num_classes = 2
num_labels = 1

title = data_name+": Full dataset"
title_X = data_name+": Features"
title_y = data_name+": Labels"

dtype_X = np.float32
atom_X = tables.Float32Atom()
dtype_y = np.uint8
atom_y = tables.UInt8Atom()


def raw_to_h5():
    '''
    Transform the raw dataset into one of HDF5 type.
    '''
    
    X_raw = np.zeros((n_all,num_features), dtype=dtype_X)
    y_raw = np.zeros((n_all,num_labels), dtype=dtype_y)
    
    print("Preparation: {}".format(data_name))

    ## Read in the raw data.
    with open(toread, newline="") as f_table:

        print("Read {}.".format(toread))
        
        f_reader = csv.reader(f_table, delimiter="\t")
        
        ## Populate the placeholder numpy arrays.
        i = 0
        for line in f_reader:
            if len(line) > 0:
                X_raw[i,:] = np.array(line[3:],
                                      dtype=X_raw.dtype)
                y_raw[i,0] = np.array(line[2],
                                      dtype=y_raw.dtype)
            i += 1
        
        ## Create and populate the HDF5 file.
        with tables.open_file(towrite, mode="w", title=title) as myh5:
            myh5.create_array(where=myh5.root,
                              name="X",
                              obj=X_raw,
                              atom=atom_X,
                              title=title_X)
            myh5.create_array(where=myh5.root,
                              name="y",
                              obj=y_raw,
                              atom=atom_y,
                              title=title_y)
            print(myh5)

        print("Wrote {}.".format(towrite))

    ## Exit all context managers before returning.
    print("Done ({}).".format(data_name))
    return None


if __name__ == "__main__":
    raw_to_h5()


###############################################################################
