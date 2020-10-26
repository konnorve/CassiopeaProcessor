#!/bin/bash
## Job name:
#SBATCH --job-name=Janis3_FFMPEG
#
## Account:
#SBATCH --account=fc_xenopus
#
## Partition:
#SBATCH --partition=savio2
#
## Wall clock limit:
#SBATCH --time=12:00:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=kve@berkeley.edu
#
## Request one node:
#SBATCH --nodes=1
#
## Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=24
#
## Processors per task:
#SBATCH --cpus-per-task=1
#
## Command(s) to run:
module load python/3.6
module load gnu-parallel/2019.03.22

source activate kve_runFFMPEG

VIDEOSCRATCHPATH=/global/scratch/kve/Janis/20200726_Janis_606pm_cam1_1
VIDEOCHUNKPATH=$VIDEOSCRATCHPATH/Video_Chunks
IMAGESTACKPATH=$VIDEOSCRATCHPATH/Image_Stacks

mkdir $IMAGESTACKPATH

ls $VIDEOCHUNKPATH/*.mp4 | parallel mkdir $IMAGESTACKPATH/{/.}
ls $VIDEOCHUNKPATH/*.mp4 | parallel ffmpeg -i {} -r 120 -q 0 $IMAGESTACKPATH/{/.}/%14d.jpg
