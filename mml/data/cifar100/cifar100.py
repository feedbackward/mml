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

data_name = "cifar100"

def get_toread_tr(num):
    return os.path.join(dir_data_toread,
                        data_name, "data_batch_{}.bin".format(num+1))

toread_tr = os.path.join(dir_data_toread,
                          data_name, "train.bin")
toread_te = os.path.join(dir_data_toread,
                          data_name, "test.bin")
newdir = os.path.join(dir_data_towrite, data_name)
towrite = os.path.join(newdir, "cifar100.h5")

n_tr = 50000
n_te = 10000
n_all = n_tr+n_te
num_features = 32*32*3
num_classes_coarse = 20
num_classes_fine = 100
num_labels = 2

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
    X_raw_te = np.zeros((n_te,num_features), dtype=dtype_X)
    y_raw_tr = np.zeros((n_tr,num_labels), dtype=dtype_y)
    y_raw_te = np.zeros((n_te,num_labels), dtype=dtype_y)
    
    print("Preparation: {}".format(data_name))
    
    ## Populate *_raw_tr.
    with open(toread_tr, mode="rb") as f_bin:
        
        print("Read {}.".format(toread_tr))
        
        for i in range(n_tr):
            
            if i % 5000 == 0:
                print("(tr) Working... image {}.".format(i))

            ## Put coarse labels in second column, fine labels in first.
            y_raw_tr[i,1] = int.from_bytes(f_bin.read(1),
                                           byteorder="big",
                                           signed=False)
            y_raw_tr[i,0] = int.from_bytes(f_bin.read(1),
                                           byteorder="big",
                                           signed=False)
            for j in range(num_features):
                X_raw_tr[i,j] = int.from_bytes(f_bin.read(1),
                                               byteorder="big",
                                               signed=False)
    
    ## Populate *_raw_te.
    with open(toread_te, mode="rb") as f_bin:
        
        print("Read {}.".format(toread_te))
        
        for i in range(n_te):
            
            if i % 1000 == 0:
                print("(te) Working... image {}.".format(i))

            ## Put coarse labels in second column, fine labels in first.
            y_raw_te[i,1] = int.from_bytes(f_bin.read(1),
                                           byteorder="big",
                                           signed=False)
            y_raw_te[i,0] = int.from_bytes(f_bin.read(1),
                                           byteorder="big",
                                           signed=False)
            for j in range(num_features):
                X_raw_te[i,j] = int.from_bytes(f_bin.read(1),
                                               byteorder="big",
                                               signed=False)
    
    ## Concatenate.
    X_raw = np.vstack((X_raw_tr, X_raw_te))
    y_raw = np.vstack((y_raw_tr, y_raw_te))
    
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
