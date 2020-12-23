#!/bin/bash

## Dataset-specific setup.
DNAME="protein"
URL="https://www.kdd.org/cupfiles/KDDCupData/2004/"
FILES_DL="data_kddcup04.tar.gz"
FILES_EX="data_kddcup04.tar.gz"

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
    echo "Extract just the guy we need from tarball, and delete original."
    for myfile in $FILES_EX
    do
	tar -f "$DIR/$myfile" --directory="$DIR" -xz "bio_train.dat"
	rm "$DIR/$myfile"
    done
fi




