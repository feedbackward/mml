'''Clerical information associated with our datasets.'''

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
                 "chance_level": 0.1,
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
    
    "iris": {"type": "classification",
             "num_classes": 3,
             "chance_level": 0.3},
    
    "mnist": {"type": "classification",
              "num_classes": 10,
              "chance_level": 0.1,
              "pix_h": 28,
              "pix_w": 28,
              "channels": 1},
    
    "protein": {"type": "classification",
                "num_classes": 2,
                "chance_level": None}
}


## List of dataset types that we allow.
allowed_types = ["classification", "regression"]


## Set default values for the fraction of data for training/validation.
_n_train_frac = 0.8 # fraction to be used for training.
_n_val_frac = 0.1*_n_train_frac # fraction to be used for validation.
for key in dataset_dict.keys():
    dataset_dict[key]["n_train_frac"] = _n_train_frac
    dataset_dict[key]["n_val_frac"] = _n_val_frac


## An alphabetically-sorted list of names whenever needed.
dataset_list = sorted([data for data in dataset_dict.keys()])


## A general-purpose data-preparation function.

def get_data_general(dataset, paras, rg, directory):
    '''
    This function assumes that "dataset" is either included in
    the dataset_dict defined here in mml.data, or that the
    accompanying "paras" argument is formatted in the same way
    as the entries in dataset_dict.
    '''

    ## Read the specified data and convert into useful dtypes.
    toread = os.path.join(directory, dataset,
                          "{}.h5".format(dataset))
    with open_file(toread, mode="r") as f:
        print(f)
        X = f.get_node(where="/", name="X").read().astype(np.float32)
        y = f.get_node(where="/", name="y").read().astype(np.int64)
        print("Types: X ({}), y ({}).".format(type(X), type(y)))
    
    ## If sample sizes are correct, then get an index for shuffling.
    n_X, num_features = X.shape
    n_y, num_labels = y.shape
    if n_X != n_y:
        s_err = "len(X) != len(y) ({} != {})".format(n_X,n_y)
        raise ValueError("Dataset sizes wrong. "+s_err)
    else:
        n_all = len(X)
        idx_shuffled = rg.permutation(n_all)

    ## Type-specific checks and modifications.
    if "type" in paras:
        if paras["type"] in allowed_types:
            if paras["type"] == "classification":
                ## All classification tasks default to one-hot labels.
                y = onehot(y=y, num_classes=paras["num_classes"])
        else:
            raise ValueError("Dataset type is not an allowed type.")
    else:
        raise ValueError("All datasets must have a type parameter.")
    
    ## Do the actual shuffling.
    X = X[idx_shuffled,:]
    y = y[idx_shuffled,:]
    
    ## Normalize the inputs in a per-feature manner.
    maxvec = X.max(axis=0,keepdims=True)
    minvec = X.min(axis=0,keepdims=True)
    X = X-minvec
    with np.errstate(divide="ignore", invalid="ignore"):
        X = X / (maxvec-minvec)
        X[X == np.inf] = 0
        X = np.nan_to_num(X)
    del maxvec, minvec

    ## Get split sizes (training, validation, testing).
    print("Shapes:")
    print("n_all =", n_all,
          "num_features =", num_features,
          "num_labels =", num_labels)
    n_train = int(n_all*paras["n_train_frac"])
    n_val = int(n_all*paras["n_val_frac"])
    n_test = n_all-n_train-n_val
    print("Subset sizes:")
    print("n_train =", n_train,
          "n_val =", n_val,
          "n_test =", n_test)
    
    ## Learning task specific parameter additions.
    paras.update({"num_features": num_features,
                  "num_labels": num_labels})

    ## Do train/test split, with validation data if specified.
    X_train = X[0:n_train,:]
    y_train = y[0:n_train,:]
    if n_val > 0:
        X_val = X[n_train:(n_train+n_val),:]
        y_val = y[n_train:(n_train+n_val),:]
    else:
        X_val = None
        y_val = None
    X_test = X[(n_train+n_val):,:]
    y_test = y[(n_train+n_val):,:]

    ## For reference, print the data types.
    print("Data types:")
    print("X_train:", type(X_train),
          "y_train:", type(y_train))
    print("X_val:", type(X_val),
          "y_val:", type(y_val))
    print("X_test:", type(X_test),
          "y_test:", type(y_test))
    
    ## Return all data, plus dataset parameters.
    return (X_train, y_train, X_val, y_val, X_test, y_test, paras)


###############################################################################


