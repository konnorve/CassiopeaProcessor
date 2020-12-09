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
TEMPOUTDIR=$VIDEOSCRATCHPATH/stdout

FRAMERATE=120

mkdir $TEMPOUTDIR
mkdir $IMAGESTACKPATH

ls $VIDEOCHUNKPATH/*.mp4 | parallel mkdir $IMAGESTACKPATH/{/.}
ls $VIDEOCHUNKPATH/*.mp4 | parallel --results $TEMPOUTDIR ffmpeg -i {} -r $FRAMERATE -q 0 $IMAGESTACKPATH/{/.}/%14d.jpg

# for individual chunks
#ffmpeg -ss 00:06:30 -i $VIDEOCHUNKPATH/xae.mp4 -r 120 -q 0 $IMAGESTACKPATH/xae/%14d.jpg
#ffmpeg -i $VIDEOCHUNKPATH/xaf.mp4 -r 120 -q 0 -t 1 $IMAGESTACKPATH/xaf/%14d.jpg