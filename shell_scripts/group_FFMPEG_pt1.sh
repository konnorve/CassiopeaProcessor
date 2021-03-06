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
#SBATCH --time=36:00:00
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
#module load python/3.6
#module load gnu-parallel/2019.03.22
#
#source activate myenv

video_home=/media/kve/DeepStorage/Sample-Data-20200726_Janis_606pm_cam1_1

for VIDEOSCRATCHPATH in $video_home
do
  VIDEOCHUNKPATH=$VIDEOSCRATCHPATH/Video_Chunks
  IMAGESTACKPATH=$VIDEOSCRATCHPATH/Image_Stacks
  TEMPOUTDIR=$VIDEOSCRATCHPATH/stdout

  FRAMERATE=120
  mkdir $VIDEOCHUNKPATH

  ls $VIDEOSCRATCHPATH/*.mp4 | parallel mv $VIDEOSCRATCHPATH/{/} $VIDEOCHUNKPATH/{/}

  mkdir $TEMPOUTDIR
  mkdir $IMAGESTACKPATH

  ls $VIDEOCHUNKPATH/*.mp4 | parallel mkdir $IMAGESTACKPATH/{/.}
  ls $VIDEOCHUNKPATH/*.mp4 | parallel --results $TEMPOUTDIR ffmpeg -i {} -r $FRAMERATE -q 0 $IMAGESTACKPATH/{/.}/%14d.jpg

done