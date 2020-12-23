#!/bin/bash

## Dataset-specific setup.
DNAME="emnist_balanced"
URL="https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
FILES_DL="gzip.zip"
FILES_EX="emnist-balanced-test-images-idx3-ubyte.gz"
FILES_EX+=" emnist-balanced-test-labels-idx1-ubyte.gz"
FILES_EX+=" emnist-balanced-train-images-idx3-ubyte.gz"
FILES_EX+=" emnist-balanced-train-labels-idx1-ubyte.gz"

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
    echo "First dealing with zip file."
    unzip -j "$DIR/$FILES_DL" "*balanced*" -d "$DIR"
    rm "$DIR/$FILES_DL"
    echo "Next decompressing the gz files."
    for myfile in $FILES_EX
    do
	gunzip "$DIR/$myfile"
    done
fi




