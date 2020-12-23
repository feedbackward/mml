#!/bin/bash

## Dataset-specific setup.
DNAME="fashion_mnist"
URL="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES_DL="train-images-idx3-ubyte.gz"
FILES_DL+=" train-labels-idx1-ubyte.gz"
FILES_DL+=" t10k-images-idx3-ubyte.gz"
FILES_DL+=" t10k-labels-idx1-ubyte.gz"
FILES_EX=$FILES_DL

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
    echo "Detected -x flag: decompressing."
    for myfile in $FILES_EX
    do
	gunzip "$DIR/$myfile"
    done
fi




