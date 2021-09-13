#!/bin/bash
## Job name:
#SBATCH --job-name=del_MarilynMonroeSD
#
## Account:
#SBATCH --account=fc_xenopus
#
## Partition:
#SBATCH --partition=savio2
#
## Wall clock limit:
#SBATCH --time=3:00:00
#
#SBATCH --mail-type=END,FAIL
#
#SBATCH --mail-user=lilianzhang@berkeley.edu
#
# Request one node:
#SBATCH --nodes=1
#
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=24
#
## Processors per task:
#SBATCH --cpus-per-task=1

module load python/3.7
module load gnu-parallel

DELETEPATH=/tmp/Image_Stacks

parallel singularity exec --no-mount tmp --overlay overlay{}.img image_latest.sif /bin/bash delfiles.sh ::: {1..24}
