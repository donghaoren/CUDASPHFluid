#!/bin/bash
#PBS -q gpu-hotel
#PBS -N ffgpu
#PBS -l nodes=1:ppn=12
#PBS -l walltime=12:00:00
#PBS -o fluid.out
#PBS -e fluid.err
#PBS -V
module load cuda
SCRATCH=/oasis/tscc/scratch/$USER/$PBS_JOBID
BUILD=/home/$USER/cudafluid
mkdir -p $SCRATCH
cd $BUILD

### Copy your profile.txt and bg.bmp to SCRATCH, like the following:
# cp profile.txt mpi_project bg*.bmp $SCRATCH

cd $SCRATCH
mkdir imgs

### Run the simulation.
### Important:
###   You have to write your own nodefile to correctly run the processes on the correct GPUs.

mpirun -np 4 ./mpi_project > project.out
