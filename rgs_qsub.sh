#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=168:00:00
#PBS -l mem=8GB
#PBS -M jure.zbontar@gmail.com
 
module purge
module load python3 

cd $HOME/devel/mc-cnn
python3 rgs_qsub.py $args
