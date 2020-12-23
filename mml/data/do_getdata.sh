#!/bin/bash

## Loop over arguments passed.

for arg
do
    echo "Doing: $arg dataset."
    bash "./$arg/getdata.sh" -x
    python "./$arg/$arg.py"
done
