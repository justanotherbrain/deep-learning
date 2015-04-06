#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=01:00:00
#PBS -l mem=18GB
#PBS -N MONKEYBUSINESS
module purge
module load torch-deps/7
module load cuda/6.5.12
DATADIR=/scratch/courses/DSGA1008/A2/binary
RESULTS=results
cd ~
mkdir -p MONKEYBUSINESS/stl10_binary
cd MONKEYBUSINESS/stl10_binary
cp -rs $DATADIR/* ./
cd ..
wget http://cims.nyu.edu/\~drg314/MONKEYBUSINESS.tgz
tar -xzf MONKEYBUSINESS.tgz
/home/drg314/torch/install/bin/th doall.lua -type cuda -size full -models 1 -loss nll -save $RESULTS -maxtime 55 -modelName combined_model.net -kaggle predictions.csv
rm -rf stl10_binary
