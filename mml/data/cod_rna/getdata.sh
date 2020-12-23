#!/bin/bash

## Dataset-specific setup.
DNAME="cod_rna"
URL="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
FILES_DL="cod-rna cod-rna.t cod-rna.r"
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




