#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -l mem=18GB
#PBS -N _MONKEYBUSINESS
module purge
module load torch-deps/7
module load cuda/6.5.12
cd $SCRATCH/A3
/home/drg314/torch/install/bin/th A3_baseline.lua 
