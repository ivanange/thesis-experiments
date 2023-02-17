#!/bin/bash
#PBS -q beta
#PBS -l select=1:ncpus=12
#PBS -N kernel_matrices
#PBS -j oe

## Use multiple of 2 with a maximum of 24 on 'ncpus' parameter, one node has 24 cores max
## With the 'select=3:ncpus=10:mpiprocs=10' option you get 30 cores on 3 nodes
## If you use select=1:ncpus=30 your job will NEVER run because no node has 30 cores.
 
# load appropriate modules
module purge
module load intel/intel-mkl-2021.3/2021.3


#move to PBS_O_WORKDIR
cd $PBS_O_WORKDIR/..
 
# Define scratch space scratchbeta on ICE XA
SCRATCH=/scratchbeta/$USER/
PROJECT='experiments'
# mkdir $SCRATCH
mkdir $SCRATCH/$PROJECT

# copy some input files to  $SCRATCH directory
cp -a .  $SCRATCH/$PROJECT

# cd $SCRATCH/$PROJECT/ivan
source conda/bin/activate thesis
cd thesis-experiments

#execute your program
## With SGI MPT use 'mpiexec_mpt -np 30 myprogram' to use mpt correctly for example

# cd $SCRATCH/$PROJECT || exit 1
python kernel_matrices.py 1> kernel_matrices.out 2> kernel_matrices.err


# copy some output files to submittion directory and delete temporary work files
cp -a kernel_matrices/. $PBS_O_WORKDIR/kernel_matrices || exit 1
 
#clean the temporary directory
rm -rf "$SCRATCH/$PROJECT"/*