#!/bin/sh

mpirun -np 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none ./main.x
