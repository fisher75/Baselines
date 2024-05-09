#!/bin/bash

### The following requests all resources on 1 DGX-1 node
#PBS -l select=1:ngpus=1:ncpus=16

### Specify amount of time required
#PBS -l walltime=20:00:00

### Specify project code
#PBS -P 12002486

### Specify name for job
#PBS -N CLIP_zeroshot_my

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### For automatic mailing, use the following options:
#PBS -M chih0001@e.ntu.edu.sg
#PBS -m abe

### Start of commands to be run
# 获取当前日期和时间，格式为 YYYYMMDD-HHMMSS
NOW=$(date +%Y%m%d-%H%M%S)

# 将标准输出和错误重定向到包含日期时间的文件名
exec > /home/users/ntu/chih0001/scratch/model/baselines/CLIP/log/${PBS_JOBNAME}_${NOW}.o${PBS_JOBID} 2>&1


source /home/users/ntu/chih0001/anaconda3/etc/profile.d/conda.sh
conda activate llava-test

export CUDA_VISIBLE_DEVICES=0

cd /home/users/ntu/chih0001/scratch/model/baselines/CLIP
python CLIP_zeroshot.py

