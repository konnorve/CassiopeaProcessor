import multiprocessing as mp
import os
from pathlib import Path
import pandas as pd
import shutil
import DataMethods as dm
import multiprocessing as mp
import os
import sys
import concurrent.futures

############################################################################################

# in the case of Savio, parent directory would be scratch/jellyname/recording name
parent_Dir = Path(sys.argv[1])

# directory for video chunks within recording
videoDir = parent_Dir / 'Video_Chunks'
stackDir = parent_Dir / 'Image_Stacks'

# home directory path for the recording
home_Dir = Path(sys.argv[2])

# Frame rate of recording
framerate = Path(sys.argv[3])


def makeOutDir(outputDir, folderName):
    outdir = outputDir / folderName
    if not outdir.exists():
        outdir.mkdir()
    return outdir

def getStackInfo(stackPath):
    name = stackPath.name
    frame_count = dm.getFrameCountFromDir_grep(stackPath)
    stackData = [parent_Dir.stem, home_Dir, name, stackPath, frame_count, framerate]
    print(stackData)
    return stackData

init_stackDir = makeOutDir(home_Dir, 'Initialization_Stack')

img_stack_dirs = dm.getSubDirectoryFilePaths(stackDir)

############################# DF Creation ######################################

initData = []

for stack in img_stack_dirs:
    initData.append(getStackInfo(stack))

# num_frames_per_chunk = [chunk.iterdir() for chunk in chunk_paths]
pre_init_DF = pd.DataFrame(initData, columns=['RecordingName',
                                            'RecordingDirPath',
                                            'ChunkName',
                                            'SavioChunkPath',
                                            'NumFramesInChunk',
                                            'FrameRate'
                                            ])

#check
print('pre init DF is')
print(pre_init_DF)

# saves pre-initialization dataframe to home directory
init_Df_Dir = makeOutDir(home_Dir, 'Initialization_DF')
pre_init_DF.to_csv(init_Df_Dir / '{}_PreInitializationDF.csv'.format(home_Dir.stem))

#check
print('DF created')
#################### copies 30s image stack ##################################

imgs_for_init = dm.getFrameFilePaths(img_stack_dirs[0])[:3600]

# check paths correct
print('image paths for 30s stack')
print(imgs_for_init[:10])

for img in imgs_for_init:
    shutil.copy(str(img), str(init_stackDir))

# check how far it got
print('image stack complete')
