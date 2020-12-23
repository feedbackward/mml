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

data_name = "adult"

toread_tr = os.path.join(dir_data_toread, data_name, "adult.data")
toread_te = os.path.join(dir_data_toread, data_name, "adult.test")
newdir = os.path.join(dir_data_towrite, data_name)
makedir_safe(newdir)
towrite = os.path.join(newdir, "adult.h5")

attribute_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
] # order is important.

attribute_dict = {
    "age": ["continuous"],
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc",
                  "Federal-gov", "Local-gov", "State-gov",
                  "Without-pay", "Never-worked"],
    "fnlwgt": ["continuous"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad",
                  "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                  "7th-8th", "12th", "Masters", "1st-4th", "10th",
                  "Doctorate", "5th-6th", "Preschool"],
    "education-num": ["continuous"],
    "marital-status": ["Married-civ-spouse", "Divorced",
                       "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service",
                   "Sales", "Exec-managerial", "Prof-specialty",
                   "Handlers-cleaners", "Machine-op-inspct",
                   "Adm-clerical", "Farming-fishing",
                   "Transport-moving", "Priv-house-serv",
                   "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family",
                     "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
             "Other", "Black"],
    "sex": ["Female", "Male"],
    "capital-gain": ["continuous"],
    "capital-loss": ["continuous"],
    "hours-per-week": ["continuous"],
    "native-country": ["United-States", "Cambodia", "England",
                       "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India",
                       "Japan", "Greece", "South", "China", "Cuba",
                       "Iran", "Honduras", "Philippines", "Italy",
                       "Poland", "Jamaica", "Vietnam", "Mexico",
                       "Portugal", "Ireland", "France",
                       "Dominican-Republic", "Laos", "Ecuador",
                       "Taiwan", "Haiti", "Columbia", "Hungary",
                       "Guatemala", "Nicaragua", "Scotland",
                       "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong",
                       "Holand-Netherlands"]
}

label_dict = {"<=50K": 0,
              ">50K": 1}

n_tr = 30162 # number of clean instances.
n_te = 15060 # number of clean instances.
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

        ## Ignore all points with missing entries.
        if value == "?":
            return (None, None)
        else:
            if num_distinct > 1:
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
        
        f_reader = csv.reader(f_table, delimiter=",",
                              skipinitialspace=True)
        
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
        
        f_reader = csv.reader(f_table, delimiter=",",
                              skipinitialspace=True)
        
        ## Populate the placeholder numpy arrays.
        idx = 0
        for i, line in enumerate(f_reader):

            if i == 0:
                continue # skip the first line, only for TEST data.
            
            if len(line) == 0:
                continue # do nothing for blank lines.
            
            ## Numpy arrays for individual instance.
            x, y = parse_line(x=line[0:-1], y=line[-1][0:-1])
            # Note: for test data, we strip training "." from labels.

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
