'''Data: Meta-data and functions for going from HDF5 to ndarray.'''

## External modules.
import numpy as np
import os
from tables import open_file

## Internal modules.
from mml.utils.linalg import onehot


###############################################################################


## Main data dictionary.
## Notes:
## - type: the type of learning problem the data is to be used for.
## - num_classes 
## - chance_level: when not None, this is #(majority class)/#(all classes).

dataset_dict = {
    "adult": {"type": "classification",
              "num_classes": 2,
              "chance_level": 0.7522},
    
    "australian": {"type": "classification",
                   "num_classes": 2,
                   "chance_level": 0.5551},
    
    "cifar10": {"type": "classification",
                "num_classes": 10,
                "chance_level": 0.1,
                "pix_h": 32,
                "pix_w": 32,
                "channels": 3},
    
    "cifar100": {"type": "classification",
                 "num_classes": 100,
                 "chance_level": None,
                 "pix_h": 32,
                 "pix_w": 32,
                 "channels": 3},
    
    "cod_rna": {"type": "classification",
                "num_classes": 2,
                "chance_level": 0.6666},
    
    "covtype": {"type": "classification",
                "num_classes": 7,
                "chance_level": None},
    
    "emnist_balanced": {"type": "classification",
                        "num_classes": 47,
                        "chance_level": 1/47,
                        "pix_h": 28,
                        "pix_w": 28,
                        "channels": 1},
    
    "fashion_mnist": {"type": "classification",
                      "num_classes": 10,
                      "chance_level": 0.1,
                      "pix_h": 28,
                      "pix_w": 28,
                      "channels": 1},

    "hills": {"type": "regression"},
    
    "iris": {"type": "classification",
             "num_classes": 3,
             "chance_level": 0.3},
    
    "mnist": {"type": "classification",
              "num_classes": 10,
              "chance_level": 0.1,
              "pix_h": 28,
              "pix_w": 28,
              "channels": 1},

    "phones": {"type": "regression"},
    
    "protein": {"type": "classification",
                "num_classes": 2,
                "chance_level": None}
}


## List of dataset types that we allow.

allowed_types = ["classification", "regression"]


## Set default values for the fraction of data for training/validation.

_n_train_frac = 0.8 # fraction to be used for training.
_n_val_frac = 0.1 # fraction to be used for validation.
for key in dataset_dict.keys():
    dataset_dict[key]["n_train_frac"] = _n_train_frac
    dataset_dict[key]["n_val_frac"] = _n_val_frac


## An alphabetically-sorted list of names whenever needed.

dataset_list = sorted([data for data in dataset_dict.keys()])


## Default numpy dtypes to use.

dtype_X = np.float32


## General-purpose data-preparation functions.

def get_data(dataset, paras, rg, directory):

    ## File to read from.
    toread = os.path.join(directory, dataset,
                          "{}.h5".format(dataset))

    ## Open the file, convert to default dtypes and do basic checks.
    with open_file(toread, mode="r") as f:
        
        print(f)
        node_list = f.list_nodes(where=f.root)

        ## Assume that all datasets include at least an "X" node.
        X = f.get_node(where=f.root, name="X").read().astype(dtype_X)
        print("Type: X ({})".format(type(X)))

        ## In addition, there can be at most one more node, called "y".
        if len(node_list) == 1:
            y = None
        elif len(node_list) == 2:
            if paras["type"] == "classification":
                dtype_y = np.int64
            elif paras["type"] == "regression":
                dtype_y = np.float32
            else:
                raise ValueError("Unknown dataset type given.")
                
            y = f.get_node(where=f.root, name="y").read().astype(dtype_y)
            print("Type: y ({})".format(type(y)))
            if len(X) != len(y):
                raise ValueError(
                    "len(X) {} != len(y) {}".format(len(X),len(y))
                )
        else:
            raise ValueError("Dataset has more than two child nodes in root.")

    return (X,y)


def get_data_general(dataset, paras, rg, directory, do_normalize=True,
                     do_shuffle=True, do_onehot=True):
    '''
    Get dataset and split into training, testing, and validation subsets.
    
    This function assumes that "dataset" is either included in
    the dataset_dict defined here in mml.data, or that the
    accompanying "paras" argument is formatted in the same way
    as the entries in dataset_dict.
    '''

    ## First get the data in ndarray form and run basic checks.
    X, y = get_data(dataset=dataset, paras=paras, rg=rg, directory=directory)
    
    ## Collect key shape information.
    n_X, num_features = X.shape
    n_y, num_labels = y.shape if y is not None else (None,None)
    n_all = n_X
    paras.update({"num_features": num_features,
                  "num_labels": num_labels})
    
    ## Carry out shuffling if prescribed.
    if do_shuffle:
        idx_shuffled = rg.permutation(n_all)
        X = X[idx_shuffled,:]
        y = y[idx_shuffled,:] if y is not None else None

    ## Type-specific checks and modifications.
    if "type" in paras:
        if paras["type"] in allowed_types:
            if paras["type"] == "classification" and do_onehot:
                ## Do one-hot labels for classification if prescribed.
                y = onehot(y=y, num_classes=paras["num_classes"])
        else:
            raise ValueError("Dataset type is not an allowed type.")
    else:
        raise ValueError("All datasets must have a type parameter.")

    ## Normalize the inputs in a per-feature manner, if prescribed.
    if do_normalize:
        maxvec = X.max(axis=0,keepdims=True)
        minvec = X.min(axis=0,keepdims=True)
        X = X-minvec
        with np.errstate(divide="ignore", invalid="ignore"):
            X = X / (maxvec-minvec)
            X[X == np.inf] = 0
            X = np.nan_to_num(X)
        del maxvec, minvec

    ## Get split sizes (training, validation, testing).
    if "n_train_frac" in paras and "n_val_frac" in paras:
        print("--Shapes--")
        print("n_all:", n_all,
              "num_features:", num_features,
              "num_labels:", num_labels)
        n_train = int(n_all*paras["n_train_frac"])
        n_val = int(n_all*paras["n_val_frac"])
        n_test = n_all-n_train-n_val
        print("--Subset sizes--")
        print("n_train:", n_train,
              "n_val:", n_val,
              "n_test:", n_test)
    else:
        raise ValueError("Dataset paras must include subset fractions.")
    
    ## Do train/test split, with validation data if specified.
    X_train = np.copy(X[0:n_train,:])
    y_train = np.copy(y[0:n_train,:]) if y is not None else None
    if n_val > 0:
        X_val = np.copy(X[n_train:(n_train+n_val),:])
        y_val = np.copy(y[n_train:(n_train+n_val),:]) if y is not None else None
    else:
        X_val = None
        y_val = None
    X_test = np.copy(X[(n_train+n_val):,:])
    y_test = np.copy(y[(n_train+n_val):,:]) if y is not None else None

    ## For reference, print the data types.
    print("Data types:")
    print("X_train:", type(X_train),
          "y_train:", type(y_train))
    print("X_val:", type(X_val),
          "y_val:", type(y_val))
    print("X_test:", type(X_test),
          "y_test:", type(y_test))
    
    ## Scrap redundant data, and return the split data, plus dataset paras.
    del X, y
    return (X_train, y_train, X_val, y_val, X_test, y_test, paras)


###############################################################################


