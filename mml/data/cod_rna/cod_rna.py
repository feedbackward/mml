'''H5 data prep'''

## External modules.
import csv
import numpy as np
import os
import tables

## Internal modules.
from mml.config import dir_data_toread
from mml.config import dir_data_towrite


###############################################################################


## Clerical setup.

data_name = "cod_rna"

toread_tr = os.path.join(dir_data_toread, data_name, "cod-rna")
toread_te = os.path.join(dir_data_toread, data_name, "cod-rna.t")
towrite = os.path.join(dir_data_towrite, data_name, "cod_rna.h5")

attribute_names = ["dynalign", "shortlen", "freqA1", "freqU1", "freqC1",
                   "freqA2", "freqU2", "freqC2"] # order is important.

attribute_dict = {
    "dynalign": ["continuous"],
    "shortlen": ["continuous"],
    "freqA1": ["continuous"],
    "freqU1": ["continuous"],
    "freqC1": ["continuous"],
    "freqA2": ["continuous"],
    "freqU2": ["continuous"],
    "freqC2": ["continuous"]
}

label_dict = {"-1": 0,
              "1": 1}

n_tr = 59535
n_te = 271617
n_all = n_tr+n_te
num_features = np.array(
    [ len(attribute_dict[key]) for key in attribute_dict.keys() ]
).sum() # number of features after a one-hot encoding.
num_classes = 2
num_labels = 1

title = data_name+": Full dataset"
title_X = data_name+": Features"
title_y = data_name+": Labels"

dtype_X = np.float32
atom_X = tables.Float32Atom()
dtype_y = np.uint8
atom_y = tables.UInt8Atom()

def parse_line(x, y):

    ## Inputs are a bit complicated.
    x_out_list = []

    for j in range(len(x)):
        
        value = x[j]
        attribute = attribute_names[j]
        num_distinct = len(attribute_dict[attribute])

        if num_distinct != 1:
            raise ValueError("Should only have one distinct value per attr.")
        else:
            wnum, numval = value.split(":")
            x_out_list.append(np.array([numval], dtype=dtype_X))
    
    x_out = np.concatenate(x_out_list)
    if len(x_out) != num_features:
        raise ValueError("Something is wrong with the feature vec parser.")

    ## Labels are easy.
    y_out = np.array([label_dict[y]], dtype=dtype_y)

    return x_out, y_out


def raw_to_h5():
    '''
    Transform the raw dataset into one of HDF5 type.
    '''
    
    X_raw_tr = np.zeros((n_tr,num_features), dtype=dtype_X)
    y_raw_tr = np.zeros((n_tr,num_labels), dtype=dtype_y)
    X_raw_te = np.zeros((n_te,num_features), dtype=dtype_X)
    y_raw_te = np.zeros((n_te,num_labels), dtype=dtype_y)
    
    print("Preparation: {}".format(data_name))
    
    ## Read in the raw training data.
    with open(toread_tr, newline="") as f_table:

        print("Read {}.".format(toread_tr))
        
        f_reader = csv.reader(f_table, delimiter=" ")
        
        ## Populate the placeholder numpy arrays.
        idx = 0
        for line in f_reader:

            if len(line) == 0:
                continue # do nothing for blank lines.

            ## Numpy arrays for individual instance.
            x, y = parse_line(x=line[1:], y=line[0])

            if x is None:
                continue # skip instances with missing values.
            else:
                X_raw_tr[idx,:] = x
                y_raw_tr[idx,0] = y

            ## Update the index (also counts the clean data points).
            idx += 1
        
        ## Check that number of *clean* instances is as expected.
        print(
            "Number of clean guys (tr): {}. Note n_tr = {}".format(idx,n_tr)
        )

    
    ## Read in the raw test data.
    with open(toread_te, newline="") as f_table:

        print("Read {}.".format(toread_te))
        
        f_reader = csv.reader(f_table, delimiter=" ")
        
        ## Populate the placeholder numpy arrays.
        idx = 0
        for i, line in enumerate(f_reader):
            
            if len(line) == 0:
                continue # do nothing for blank lines.
            
            ## Numpy arrays for individual instance.
            x, y = parse_line(x=line[1:], y=line[0])

            if x is None:
                continue # skip instances with missing values.
            else:
                X_raw_te[idx,:] = x
                y_raw_te[idx,0] = y

            ## Update the index (also counts the clean data points).
            idx += 1
        
        ## Check that number of *clean* instances is as expected.
        print(
            "Number of clean guys (te): {}. Note n_te = {}".format(idx,n_te)
        )
    
    
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
