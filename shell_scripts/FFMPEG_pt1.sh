#!/bin/bash
## Job name:
#SBATCH --job-name=pink_FFMPEG_test0
#
## Account:
#SBATCH --account=fc_xenopus
#
## Partition:
#SBATCH --partition=savio2
#
## Wall clock limit:
#SBATCH --time=00:30:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=lilianzhang@berkeley.edu
#
##  Number of MPI tasks needed:
#SBATCH --ntasks=25
#
## Processors per task:
#SBATCH --cpus-per-task=1
#
## Command(s) to run:
module load python/3.6
module load gnu-parallel/2019.03.22

source activate kve_runFFMPEG

VIDEOSCRATCHPATH=/global/scratch/kve/Lgaga/20200723_Lgaga_730pm_cam2_1
VIDEOCHUNKPATH=$VIDEOSCRATCHPATH/Video_Chunks
IMAGESTACKPATH=$VIDEOSCRATCHPATH/Image_Stacks

$FRAMERATE=120

mkdir $IMAGESTACKPATH
ls $VIDEOCHUNKPATH/*.mp4 | parallel mkdir $IMAGESTACKPATH/{/.}
ls $VIDEOCHUNKPATH/*.mp4 | parallel ffmpeg -i {} -r $FRAMERATE -q 0 $IMAGESTACKPATH/{/.}/%14d.jpg
