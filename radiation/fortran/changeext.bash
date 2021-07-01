#!/bin/bash

for file in *.f; do 
    mv -- "$file" "${file%.f}.F"
done
