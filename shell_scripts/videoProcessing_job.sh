#!/bin/bash
# Job name: ffmpeg_test
#SBATCH --job-name=VP_MarilynMonroe_Baseline
#
# Account:
#SBATCH --account=fc_xenopus
#
# Partition:
#SBATCH --partition=savio_bigmem
#
#
# Wall clock limit:
#SBATCH --time=36:00:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=lilianzhang@berkeley.edu
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

#source activate myenv

# Updating below paths lines not necessary, but good for keeping track
JELLYPATH=/global/scratch/users/lilianzhang/RNASeq2/20210702/MarilynMonroe/Baseline/
POSTINIT_DF_PATH=/global/scratch/users/lilianzhang/RNASeq2/20210702/MarilynMonroe/SD/MarilynMonroeSD_PostInitializationDF.csv
PARENTDIR=/tmp/vp_data

parallel singularity exec --no-mount tmp --overlay overlay{}.img image_latest.sif /bin/bash vp_exec_script.sh ::: {1..24}
