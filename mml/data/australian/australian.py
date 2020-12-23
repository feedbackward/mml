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

data_name = "australian"

toread = os.path.join(dir_data_toread, data_name, "australian.dat")
newdir = os.path.join(dir_data_towrite, data_name)
makedir_safe(newdir)
towrite = os.path.join(newdir, "australian.h5")

attribute_names = ["A"+str(i+1) for i in range(14)] # order is important.

attribute_dict = {
    "A1": [str(i) for i in range(2)],
    "A2": ["continuous"],
    "A3": ["continuous"],
    "A4": [str(i+1) for i in range(3)],
    "A5": [str(i+1) for i in range(14)],
    "A6": [str(i+1) for i in range(9)],
    "A7": ["continuous"],
    "A8": [str(i) for i in range(2)],
    "A9": [str(i) for i in range(2)],
    "A10": ["continuous"],
    "A11": [str(i) for i in range(2)],
    "A12": [str(i+1) for i in range(3)],
    "A13": ["continuous"],
    "A14": ["continuous"]
}

label_dict = {"0": 0,
              "1": 1}

n_all = 690
# note: Quinlan's UCI memo says there are missing values, but looking
#       at the data and online refs, it seems no values are missing.
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

    ## Inputs are just a bit complicated.
    x_out_list = []

    for j in range(len(x)):
        
        value = x[j]
        attribute = attribute_names[j]
        num_distinct = len(attribute_dict[attribute])

        if num_distinct > 1:
            idx_hot = attribute_dict[attribute].index(value)
            onehot = np.zeros(num_distinct, dtype=dtype_X)
            onehot[idx_hot] = 1.0
            x_out_list.append(onehot)
        else:
            x_out_list.append(np.array([x[j]], dtype=dtype_X))
    
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
    
    X_raw = np.zeros((n_all,num_features), dtype=dtype_X)
    y_raw = np.zeros((n_all,num_labels), dtype=dtype_y)
    
    print("Preparation: {}".format(data_name))
    
    ## Read in the raw data.
    with open(toread, newline="") as f_table:
        
        print("Read {}.".format(toread))
        
        f_reader = csv.reader(f_table, delimiter=" ")
        
        ## Populate the placeholder numpy arrays.
        idx = 0
        for line in f_reader:
            
            if len(line) == 0:
                continue # do nothing for blank lines.
            
            ## Numpy arrays for individual instance.
            x, y = parse_line(x=line[0:-1], y=line[-1])

            if x is None:
                continue # skip instances with missing values.
            else:
                X_raw[idx,:] = x
                y_raw[idx,0] = y

            ## Update the index (also counts the clean data points).
            idx += 1
        
        ## Check that number of *clean* instances is as expected.
        print(
            "Number of clean guys: {}. Note n_all = {}".format(idx,n_all)
        )
        

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
