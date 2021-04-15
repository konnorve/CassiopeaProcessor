#!/bin/bash
# Job name:
#SBATCH --job-name=cell_pro
#
# Account:
#SBATCH --account=fc_xenopus
#
# Partition:
#SBATCH --partition=savio2_htc
#
# Wall clock limit:
#SBATCH --time=10:00:00
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

EXPERIMENTPATH=/global/scratch/kve/Proliferation_experiment_1


for JELLY in $EXPERIMENTPATH/*/

do
	echo $JELLY
	for VIDEOSCRATCHPATH in $JELLY/*/
	do
		echo $VIDEOSCRATCHPATH
		HOMEDIRPATH=$VIDEOSCRATCHPATH
		FRAMERATE=120

		python3 /global/home/groups/fc_xenopus/utils/CassiopeaProcessor/FFMPEG_p2.py $VIDEOSCRATCHPATH $HOMEDIRPATH $FRAMERATE

	done
done
