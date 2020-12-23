'''H5 data prep'''

## External modules.
import numpy as np
import os
import tables

## Internal modules.
from mml.config import dir_data_toread
from mml.config import dir_data_towrite
from mml.utils import makedir_safe


###############################################################################


## Clerical setup.

data_name = "fashion_mnist"

toread_X_tr = os.path.join(dir_data_toread,
                            data_name, "train-images-idx3-ubyte")
toread_X_te = os.path.join(dir_data_toread,
                            data_name, "t10k-images-idx3-ubyte")
toread_y_tr = os.path.join(dir_data_toread,
                            data_name, "train-labels-idx1-ubyte")
toread_y_te = os.path.join(dir_data_toread,
                            data_name, "t10k-labels-idx1-ubyte")
newdir = os.path.join(dir_data_towrite, data_name)
makedir_safe(newdir)
towrite = os.path.join(newdir, "fashion_mnist.h5")

n_tr = 60000
n_te = 10000
n_all = n_tr+n_te
num_features = 28*28
num_classes = 10
num_labels = 1

title = data_name+": Full dataset"
title_X = data_name+": Features"
title_y = data_name+": Labels"

dtype_X = np.uint8
atom_X = tables.UInt8Atom()
dtype_y = np.uint8
atom_y = tables.UInt8Atom()


def raw_to_h5():
    '''
    Transform the raw dataset into one of HDF5 type.
    '''
    
    X_raw_tr = np.zeros((n_tr,num_features), dtype=dtype_X)
    y_raw_tr = np.zeros((n_tr,num_labels), dtype=dtype_y)
    X_raw_te = np.zeros((n_te,num_features), dtype=dtype_X)
    y_raw_te = np.zeros((n_te,num_labels), dtype=dtype_y)
    
    print("Preparation: {}".format(data_name))
    
    ## Populate X_raw_tr.
    with open(toread_X_tr, mode="rb") as f_bin:
        
        print("Read {}.".format(toread_X_tr))
        f_bin.seek(16) # go to start of images.

        for i in range(n_tr):
            
            if i % 5000 == 0:
                print("(tr) Working... image {}.".format(i))

            for j in range(num_features):
                X_raw_tr[i,j] = int.from_bytes(f_bin.read(1),
                                               byteorder="big",
                                               signed=False)
    
    ## Populate X_raw_te.
    with open(toread_X_te, mode="rb") as f_bin:

        print("Read {}.".format(toread_X_te))
        f_bin.seek(16) # go to start of images.
        
        for i in range(n_te):
            
            if i % 1000 == 0:
                print("(te) Working... image {}.".format(i))
                
            for j in range(num_features):
                X_raw_te[i,j] = int.from_bytes(f_bin.read(1),
                                               byteorder="big",
                                               signed=False)
    
    ## Populate y_raw_tr.
    with open(toread_y_tr, mode="rb") as f_bin:

        print("Read {}.".format(toread_y_tr))
        f_bin.seek(8) # go to start of labels.
        
        for i in range(n_tr):
            y_raw_tr[i,0] = int.from_bytes(f_bin.read(1),
                                           byteorder="big",
                                           signed=False)

    ## Populate y_raw_te.
    with open(toread_y_te, mode="rb") as f_bin:

        print("Read {}.".format(toread_y_te))
        f_bin.seek(8) # go to start of labels.
        
        for i in range(n_te):
            y_raw_te[i,0] = int.from_bytes(f_bin.read(1),
                                           byteorder="big",
                                           signed=False)


    ## Concatenate.
    X_raw = np.vstack((X_raw_tr, X_raw_te))
    y_raw = np.vstack((y_raw_tr, y_raw_te))
    
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
