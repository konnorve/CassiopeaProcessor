#!/bin/bash
# Job name:
#SBATCH --job-name=LGAGA_InitDF
#
# Account:
#SBATCH --account=fc_xenopus
#
# Partition:
#SBATCH --partition=savio2_htc
#
# Wall clock limit:
#SBATCH --time=03:00:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=lilianzhang@berkeley.edu
#
## Request one node:
#SBATCH --nodes=1
#
## Processors per task:
#SBATCH --cpus-per-task=1
#
## Command(s) to run:
#module load python/3.6
#module load gnu-parallel/2019.03.22
#source activate myenv

VIDEOSCRATCHPATH=/media/kve/DeepStorage/Sample-Data-20200726_Janis_606pm_cam1_1
HOMEDIRPATH=$VIDEOSCRATCHPATH
FRAMERATE=120

# python3 /global/home/groups/fc_xenopus/utils/CassiopeaProcessor/FFMPEG_p2.py $VIDEOSCRATCHPATH $HOMEDIRPATH $FRAMERATE

python3 /home/kve/PycharmProjects/CassiopeaProcessor/FFMPEG_p2.py $VIDEOSCRATCHPATH $HOMEDIRPATH $FRAMERATE