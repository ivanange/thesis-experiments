#!/bin/bash
#PBS -q beta
#PBS -l select=1:ncpus=12
#PBS -N sift_rank_log
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
python experimenter.py train sift log --rank_ratio=1 1> sift_rank_log.out 2> sift_rank_log.err


# copy some output files to submittion directory and delete temporary work files
cp -a models/. $PBS_O_WORKDIR/models || exit 1
 
#clean the temporary directory
rm -rf "$SCRATCH/$PROJECT"/*