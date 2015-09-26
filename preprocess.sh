#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=1:00:00
#PBS -l mem=32GB
#PBS -M jure.zbontar@gmail.com
 
module purge
module load numpy 
module load opencv

export TERM=xterm

cd $HOME/devel/mc-cnn
./preprocess_mb.py $args
