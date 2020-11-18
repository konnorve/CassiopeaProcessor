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
#SBATCH --time=00:30:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=kve@berkeley.edu
#
## Request one node:
#SBATCH --nodes=1
#
## Processors per task:
#SBATCH --cpus-per-task=1
#
## Command(s) to run:
module load python/3.6
module load gnu-parallel/2019.03.22
source activate CassiopeaProcessor

VIDEOSCRATCHPATH=/global/scratch/kve/Janis/20200726_Janis_606pm_cam1_1
HOMEDIRPATH=/global/home/kve/Janis/20200726_Janis_606pm_cam1_1

python3 /global/home/groups/fc_xenopus/utils/CassiopeaProcessor/FFMPEG_p2.py $VIDEOSCRATCHPATH $HOMEDIRPATH


