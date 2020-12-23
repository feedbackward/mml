#!/bin/bash

## Dataset-specific setup.
DNAME="iris"
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"
FILES_DL="bezdekIris.data iris.data iris.names"
FILES_EX=""

## Generic setup.
DIR=$(python config_get.py $DNAME)
echo $DIR

## Based on setup, downloads everything specified.
echo "Starting download via wget."
for myfile in $FILES_DL
do
    wget --directory-prefix=$DIR "$URL$myfile"
done


## If flagged, then expand/decompress as well.
if [ "$1" = "-x" ]
then
    echo "Nothing to be de-compressed for this data set."
fi




