#!/bin/bash
# Job name: ffmpeg_test
#SBATCH --job-name=VPM_lg1_kve
#
# Account:
#SBATCH --account=fc_xenopus
#
# Partition:
#SBATCH --partition=savio2_htc
#
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=kve@berkeley.edu
#
# Request one node:
#SBATCH --nodes=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
## Command(s) to run:
module load gcc openmpi
module load python
module load gnu-parallel/2019.03.22
source activate CassiopeaProcessor
parallel python3 /global/home/groups/fc_xenopus/utils/CassiopeaProcessor-FinalTest/VideoProcessor_Main.py ::: 0
