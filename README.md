# mml: base software for ML prototyping

This repository is a collection of general-purpose software that we (<a href="https://github.com/feedbackward">M.J. Holland</a> and colleages) use to implement, test, and disseminate machine learning models and algorithms that are developed as part of our ongoing research activities.

We call it "base" software since it plays a supporting role in our research projects. Functions and classes that are used frequently across different distinct projects will be implemented and maintained here, in contrast with project-specific repositories, which contain only the code that is very "local" to each project (see for example <a href="https://github.com/feedbackward/sgd-roboost">sgd-roboost</a> or <a href="https://github.com/feedbackward/robrisk">robrisk</a>).

This README file is composed of the following content:

- <a href="#prereq">Prerequisites</a>
- <a href="#start">Getting started: example of expected usage</a>
- <a href="#data">Acquiring benchmark datasets</a>
- <a href="#content_overview">Overview of repository contents</a>


<a id="prereq"></a>
## Prerequisites

For the purposes of this documentation, regarding the user's environment we will assume that they

- have access to a `bash` shell
- can use `wget` to download data sets
- have `unzip`, `git`, and `conda` installed

and finally that they have run

```
$ conda update -n base conda
```

in order to bring `conda` up to date. Experienced users can freely swap out any of the above utilities with alternatives of their choice; just keep in mind that the documentation is based on the above environment.


<a id="start"></a>
## Getting started: example of expected usage

As mentioned above, this is "base" code, intended to be used in a cohesive manner with other repositories, containing code that is more "local" to specific projects. Below, we illustrate a simple expected use case.

```
$ git clone https://github.com/feedbackward/[project name].git
$ git clone https://github.com/feedbackward/mml.git
$ conda create -n [project name] python=3.8 jupyter matplotlib pip pytables scipy unzip
$ conda activate [project name]
([project name]) $ cd mml
([project name]) $ pip install -e ./
```

That is, we `clone` both the base and project-local software repositories, create a project-specific virtual environment with the required additional software, and then install `mml` for easy importing on the fly; the option `-e` installs the package in edit mode, so that changes made locally are reflected immediately, without the need to re-install every time a change is made. In practice, of course, the placeholder `[project name]` will be replaced with the appropriate project name (such as `sgd-roboost` or `robrisk`, etc.).

In the event that the latest version of `mml` becomes no longer compatible with a particular project repository, one can just use a past version of `mml` that has been verified to work with the project of interest. To do this, one would just run

```
([project name]) $ git checkout [safe hash]
([project name]) $ pip install -e ./
```

where the `[safe hash]` placeholder refers to a specific SHA-1 hash value associated with a `git` commit of `mml` that has been tested by the owner and is known to work with the project-local software of interest. For example, refer to the safe hashes documented for <a href="https://github.com/feedbackward/sgd-roboost#safehash">sgd-roboost</a> and <a href="https://github.com/feedbackward/robrisk#safehash">robrisk</a>.


<a id="data"></a>
## Acquiring benchmark datasets

Preparing datasets in a standardized format (HDF5) is extremely easy using the scripts housed in `mml/data`. Assuming the context of the <a href="#start">previous sub-section</a>, then all we need to run is

```
([project name]) $ cd [path to mml]/mml/data
([project name]) $ bash do_getdata.sh [dataset1 dataset2 ...]
```

where the `dataset*` arguments can be the name of any directory included in `mml/data` (e.g., `adult`, `australian`, `cifar10`, and so forth). The raw data retains the original file names, while the processed data is all named `[dataset].h5`.

Regarding where the data is stored, the default behaviour is to store both the raw data and the processed HDF5 file `[dataset].h5` in the same directory as the data-fetching scripts, namely `mml/data/[dataset]`. If this is inconvenient for you, feel free to modify where things are stored by adjusting the following variables in `mml/config.py` manually:

- `dir_data_toread`: this is where the raw data files downloaded over the internet will be stored.
- `dir_data_towrite`: this is where the processed data files (all in .h5 format) will be stored.


<a id="content_overview"></a>
## Overview of repository contents

Here we give an overview of the main contents of this repository, stored in the `mml` directory. We do not list every file, but we cover all the key components.

- `algos/`: algorithm class definitions.

  - `gd.py`: traditional gradient descent.
  - `__init__.py`: algorithm base class definitions.
  - `linesearch.py`: base class for line search algorithms.
  - `rgd.py`: robust gradient descent.

- `config.py`: main configuration file.

- `data/`: all the sub-directories of `data` correspond to one and only one dataset. Each includes scripts for acquiring raw data and converting the raw data into a standardized HDF5 format.

  - `__init__.py`: general-purpose functions for going from `.h5` to `ndarray`, plus all relevant "meta-data" for each dataset.
  - `adult/`: the adult census data set for predicting annual income.
  - `australian/`: Australian credit data.
  - `cifar10/`: CIFAR-10 tiny images.
  - `cifar100/`: CIFAR-100 tiny images.
  - `cod_rna/`: RNA coding dataset.
  - `covtype/`: predicting forest cover type from cartographic variables.
  - `data_list.txt`: a line-broken list of all the datasets we have prepared.
  - `emnist_balanced/`: balanced extended+modified NIST dataset.
  - `fashion_mnist/`: ten classes of clothing items in MNIST digit style.
  - `iris/`: Fisher's Iris data set.
  - `mnist/`: MNIST handwritten digits.
  - `protein/`: protein homology dataset.

- `losses/`: loss class definitions.

  - `absolute.py`: absolute difference.
  - `classification.py`: penalties for classifiers (zero-one etc.).
  - `cvar.py`: CVaR loss wrapper.
  - `__init__.py`: base loss class definitions.
  - `logistic.py`: logistic loss (arbitrary number of classes).
  - `quadratic.py`: squared error.
  

- `models/`: model class definitions.

  - `__init__.py`: base model class definitions.
  - `linreg.py`: linear regressors (both single and multiple output).

- `utils/`

  - `__init__.py`: houses a variety of helper functions, mostly of a clerical nature.
  - `linalg.py`: helper functions related to array manipulation.
  - `mest.py`: various helper functions related to M-estimation.
  - `rgen.py`: random data generation based on modern `numpy.random.Generator` objects.
  - `vecmean.py`: a collection of vector mean estimation routines.


It is also worth mentioning that in the top level of this repository, we have the following additional documentation:

- `refs_mest.md`: a simple bibliography of references cited in `mml/utils/mest.py`.
- `refs_rgen.md`: a concise list of numpy and scipy references relevant to the data-generating and statistic-producing functions implemented in `mml/utils/rgen.py`.
