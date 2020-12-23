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

data_name = "covtype"

toread = os.path.join(dir_data_toread, data_name, "covtype.data")
newdir = os.path.join(dir_data_towrite, data_name)
towrite = os.path.join(newdir, "covtype.h5")

label_dict = {"Spruce-Fir": 0,
              "Lodgepole Pine": 1,
              "Ponderosa Pine": 2,
              "Cottonwood-Willow": 3,
              "Aspen": 4,
              "Douglas Fir": 5,
              "Krummholz": 6} # these labels are after our shifting.

n_all = 581012
num_features = 54
num_classes = 7
num_labels = 1

title = data_name+": Full dataset"
title_X = data_name+": Features"
title_y = data_name+": Labels"

dtype_X = np.int16
atom_X = tables.Int16Atom()
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
        
        f_reader = csv.reader(f_table, delimiter=",")
        
        ## Populate the placeholder numpy arrays.
        i = 0
        for line in f_reader:
            if len(line) > 0:
                X_raw[i,:] = np.array(line[0:-1],
                                      dtype=X_raw.dtype)
                y_raw[i,0] = np.array(line[-1],
                                      dtype=y_raw.dtype)-1 # subtraction.
            i += 1
        
        ## Create and populate the HDF5 file.
        makedir_safe(newdir)
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
