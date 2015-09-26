#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l mem=24GB
#PBS -j oe
#PBS -M jure.zbontar@gmail.com
 
module purge
module load cuda
module load cudnn
module load opencv
module load python3 

cd $HOME/devel/mc-cnn
./hs.py $args
