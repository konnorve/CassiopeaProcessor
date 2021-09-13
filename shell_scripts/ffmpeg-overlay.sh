#!/bin/bash

## Job name:
#SBATCH --job-name=FFMPEG_MariahBaseline
#
## Account:
#SBATCH --account=fc_xenopus
#
## Partition:
#SBATCH --partition=savio2
#
## Wall clock limit:
#SBATCH --time=5:00:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=lilianzhang@berkeley.edu
#
# Request one node:
#SBATCH --nodes=1
#
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=24
#
## Processors per task:
#SBATCH --cpus-per-task=1
#
## Command(s) to run:
module load python/3.6
module load gnu-parallel/2019.03.22

VIDEOSCRATCHPATH=/global/scratch/users/lilianzhang/RNASeq2/20210621/Mariah/Baseline
VIDEOCHUNKPATH=$VIDEOSCRATCHPATH/Video_Chunks

ls $VIDEOCHUNKPATH/*.mp4 | parallel singularity exec --no-mount tmp --overlay overlay{%}.img ffmpeg_latest.sif /bin/bash exec_script.sh {} {/.}
