#!/bin/bash
# Job name: ffmpeg_test
#SBATCH --job-name=VPlgaga1_kve
#
# Account:
#SBATCH --account=fc_xenopus
#
# Partition:
#SBATCH --partition=savio2
#
#
# Wall clock limit:
#SBATCH --time=48:00:00
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

POSTINIT_DF_PATH=/global/scratch/kve/Janis/20200726_Janis_606pm_cam1_1
PARENTDIR="$(dirname "$POSTINIT_DF_PATH")"
TEMPOUTDIR=$PARENTDIR/stdout

mkdir $TEMPOUTDIR

parallel --results $TEMPOUTDIR python3 /global/home/groups/fc_xenopus/utils/CassiopeaProcessor-FinalTest/VideoProcessor_Main.py ::: $POSTINIT_DF_PATH ::: $(seq 60)

