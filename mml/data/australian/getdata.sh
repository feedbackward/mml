#!/bin/bash

## Dataset-specific setup.
DNAME="australian"
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/"
FILES_DL="australian.dat australian.doc"
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
    echo "Detected -x flag."
    echo "Nothing to be done this time, exiting."
fi




