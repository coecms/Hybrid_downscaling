#!/bin/bash

#PBS -P vp91
#PBS -q gpuvolta
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=120GB
#PBS -l jobfs=100GB
#PBS -l walltime=1:15:00
#PBS -l wd
#PBS -l storage=gdata/hh5+scratch/vp91

### REQUIRES: ###
eval "$(conda shell.bash hook)"
conda activate /g/data3/hh5/public/apps/miniconda3/envs/pytorch-gpu6

cd /scratch/vp91/CLEX/

python main_MLP_accelerated.py
echo "bye"
