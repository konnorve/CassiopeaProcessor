import multiprocessing as mp
import os
from pathlib import Path
import pandas as pd
import shutil
import DataMethods as dm

import concurrent.futures

############################################################################################

# in the case of Savio, parent directory would be scratch/jellyname/recording name
parent_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/Short_Behavioral_Recordings/Scratch/NinaSimone/')

# parent_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/PinkTrainingData_Scratch')

# directory for video chunks within recording
videoDir = parent_Dir / 'Video_Chunks'
stackDir = parent_Dir / 'Image_Stacks'

# home directory path for the recording
home_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/Short_Behavioral_Recordings/Home/NinaSimone')
# home_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/PinkTrainingData_Home')

# Frame rate of recording
framerate = 120


def makeOutDir(outputDir, folderName):
    outdir = outputDir / folderName
    if not outdir.exists():
        outdir.mkdir()
    return outdir

init_stackDir = makeOutDir(home_Dir, 'Initialization_Stack')

img_stack_dirs = dm.getSubDirectoryFilePaths(stackDir)

############################# DF Creation ######################################

initData = []

def getStackInfo(stackPath):
    name = stackPath.name
    frame_count = dm.getFrameCountFromDir(stackPath)
    stackData = [parent_Dir.stem, home_Dir, name, stackPath, frame_count]
    print(stackData)
    initData.append(stackData)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(getStackInfo, img_stack_dirs)

# num_frames_per_chunk = [chunk.iterdir() for chunk in chunk_paths]
pre_init_DF = pd.DataFrame(initData, header=[
                                            'RecordingName',
                                            'RecordingDirPath',
                                            'ChunkName',
                                            'SavioChunkPath',
                                            'NumFramesInChunk'
                                            ])

pre_init_DF['FrameRate'] = framerate

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
    print(shutil.copy(str(img), str(init_stackDir)))

# check how far it got
print('image stack complete')
