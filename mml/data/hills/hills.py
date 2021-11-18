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

data_name = "hills"

toread = os.path.join(dir_data_toread, data_name, "hills.csv")
newdir = os.path.join(dir_data_towrite, data_name)
towrite = os.path.join(newdir, "hills.h5")

attribute_names = [ "name", "dist", "climb" ] # order is important.

    

attribute_dict = {
    "name": ["Greenmantle", "Carnethy", "Craig Dunain", "Ben Rha",
             "Ben Lomond", "Goatfell", "Bens of Jura", "Cairnpapple",
             "Scolty", "Traprain", "Lairig Ghru", "Dollar", "Lomonds",
             "Cairn Table", "Eildon Two", "Cairngorm", "Seven Hills",
             "Knock Hill", "Black Hill", "Creag Beag", "Kildcon Hill",
             "Meall Ant-Suidhe", "Half Ben Nevis", "Cow Hill",
             "N Berwick Law", "Creag Dubh", "Burnswark", "Largo Law",
             "Criffel", "Acmony", "Ben Nevis", "Knockfarrel",
             "Two Breweries", "Cockleroi", "Moffat Chase"],
    "dist": ["continuous"],
    "climb": ["continuous"]
}

n_all = 35
num_features = 35+2
num_classes = None # since regression, set to None.
num_labels = 1

title = data_name+": Full dataset"
title_X = data_name+": Features"
title_y = data_name+": Labels"

dtype_X = np.float32
atom_X = tables.Float32Atom()
dtype_y = np.float32
atom_y = tables.Float32Atom()

def parse_line(x, y):

    ## Inputs are a bit complicated.
    x_out_list = []

    for j in range(len(x)):
        
        value = x[j]
        attribute = attribute_names[j]
        num_distinct = len(attribute_dict[attribute])

        if num_dictinct > 1:
            idx_hot = attribute_dict[attribute].index(value)
            onehot = np.zeros(num_distinct, dtype=dtype_X)
            onehot[idx_hot] = 1.0
            x_out_list.append(onehot)
        else:
            x_out_list.append(np.array([value], dtype=dtype_X))

    x_out = np.concatenate(x_out_list)
            
    if len(x_out) != num_features:
        raise ValueError("Something is wrong with the feature vec parser.")
    
    ## Labels are easy.
    y_out = np.array(y, dtype=dtype_y)
    
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
        
        f_reader = csv.reader(f_table, delimiter=",",
                              skipinitialspace=True)
        
        ## Populate the placeholder numpy arrays.
        idx = 0
        for linenum, line in enumerate(f_reader):
            
            if linenum == 0:
                continue # skip the first row.
            
            if len(line) == 0:
                continue # do nothing for blank lines.
            
            ## Numpy arrays for individual instance.
            x, y = parse_line(x=line[0:-1], y=line[-1])
            X_raw[idx,:] = x
            y_raw[idx,0] = y
            
            ## Update the index.
            idx += 1
        
        ## Check that number of *clean* instances is as expected.
        print(
            "Number of clean guys: {}. Note n_all = {}".format(idx,n_all)
        )


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
