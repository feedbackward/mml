#!/bin/bash

## Dataset-specific setup.
DNAME="cifar100"
URL="http://www.cs.toronto.edu/~kriz/"
FILES_DL="cifar-100-binary.tar.gz"
FILES_EX="cifar-100-binary.tar.gz"

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
    echo "Extract from tarball, decompress, and delete original."
    for myfile in $FILES_EX
    do
	tar -xzf "$DIR/$myfile" --directory="$DIR" --strip=1
	rm "$DIR/$myfile"
	rm "$DIR/fine_label_names.txt~"
    done
fi




