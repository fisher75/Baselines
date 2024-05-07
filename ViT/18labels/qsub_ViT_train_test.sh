#!/bin/bash

### The following requests all resources on 1 DGX-1 node
#PBS -l select=1:ngpus=1:ncpus=16

### Specify amount of time required
#PBS -l walltime=20:00:00

### Specify project code
#PBS -P 12002486

### Specify name for job
#PBS -N ViT_train

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### For automatic mailing, use the following options:
#PBS -M chih0001@e.ntu.edu.sg
#PBS -m abe

### Start of commands to be run
source /home/users/ntu/chih0001/anaconda3/etc/profile.d/conda.sh
conda activate llava

export CUDA_VISIBLE_DEVICES=0

cd /home/users/ntu/chih0001/scratch/model/baselines/ViT
python various-models-exp-test.py

