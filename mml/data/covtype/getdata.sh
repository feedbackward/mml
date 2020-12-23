#!/bin/bash

## Dataset-specific setup.
DNAME="covtype"
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/"
FILES_DL="covtype.data.gz covtype.info"
FILES_EX="covtype.data.gz"

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
    echo "Decompressing the gz file."
    for myfile in $FILES_EX
    do
	gunzip "$DIR/$myfile"
    done
fi




