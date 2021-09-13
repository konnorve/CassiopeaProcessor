#!/bin/bash

POSTINIT_DF_PATH=/global/scratch/users/lilianzhang/RNASeq2/20210702/MarilynMonroe/SD/MarilynMonroeSD_PostInitializationDF.csv
TEMP_IMG_STACKS=/tmp/Image_Stacks

python3 /global/scratch/users/lilianzhang/CassiopeaProcessor_20210804/VideoProcessor_Main.py $POSTINIT_DF_PATH $TEMP_IMG_STACKS
