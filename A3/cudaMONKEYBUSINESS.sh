#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=06:00:00
#PBS -l mem=70GB
#PBS -N cuda
module purge
module load torch-deps/7
module load cuda/6.5.12
cd $SCRATCH/A3
/home/drg314/torch/install/bin/th MONKEY_BUSINESS_main.lua -debug 0 -maxTime 360 -sentenceDim 100 -padding 5 -type cuda
