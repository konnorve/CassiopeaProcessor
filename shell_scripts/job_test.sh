#!/bin/bash
# Job name: ffmpeg_test
#SBATCH --job-name=ffmpeg_20201006
#
# Account:
#SBATCH --account=fc_xenopus
#
# Partition:
#SBATCH --partition=savio2
#
#
# Wall clock limit:
#SBATCH --time=00:30:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=kve@berkeley.edu
#
# Request one node:
#SBATCH --nodes=1
#
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=24
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
## Command(s) to run:
module load gcc openmpi
module load python
source activate CassiopeaProcessor
python3 /global/home/groups/fc_xenopus/utils/CassiopeaProcessor-FinalTest/MultiProcess_FFMPEG.py
