#!/bin/bash
# Job name:
#SBATCH --job-name=LGAGA_FFMPEG
#
# Account:
#SBATCH --account=fc_xenopus
#
# Partition:
#SBATCH --partition=savio2
#
# Wall clock limit:
#SBATCH --time=10:00:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=lilianzhang@berkeley.edu
#
## Command(s) to run:
module load python/3.6
source activate myenv
python3 /global/home/groups/fc_xenopus/utils/CassiopeaProcessor-FinalTest/MultiProcess_FFMPEG.py
