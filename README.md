# mml: base software for ML prototyping

This repository is a collection of general-purpose software that we use for developing and testing machine learning models and algorithms, as part of the research conducted by the repository owner (Matthew J. Holland) and colleagues.

We call it "base" software since this repository plays an ancillary role in our research projects. Functions and classes that are used frequently in different settings will be implemented and maintained here, in contrast with project-specific repositories, which contain only the code that is very "local" to each project.

Instead of developing everything behind closed doors and then taking a small slice of that code public in a polished demo, in the aim of increased transparency and easier reproducability, we are adopting a *"designed to go public"* approach to our experimental methodology.


## Getting started

It is assumed that the user has access to a `bash` shell, can use `wget` to download data sets, has `git` and `conda` installed, and has run the following command before starting:

```
$ conda update -n base conda
```

As mentioned above, this is "base" code, intended to be used in a cohesive manner with other repositories, containing code that is more "local" to specific projects. Below, we illustrate a simple expected use case.

```
$ git clone https://github.com/feedbackward/[project name].git
$ git clone https://github.com/feedbackward/mml.git
$ conda create -n [project name] python=3.8 jupyter matplotlib pip pytables scipy
$ conda activate [project name]
$ ([project name]) cd mml
$ ([project name]) git checkout [SHA-1]
$ ([project name]) pip install -e ./
```

That is, we `clone` both the base and project-local software repositories, create a project-specific environment with standard software, and then install `mml` for easy importing by project repositories. The `[SHA-1]` part refers to a specific SHA-1 hash value associated with a particular `git` commit. Depending on the project, the `git checkout` line may be unnecessary; it will be used to ensure that even as `mml` gets updated, the project specific code uses a version of `mml` that it works with. The option `-e` installs the package in edit mode, so that changes made locally are reflected immediately, without the need to re-install.

