import multiprocessing as mp
import os
from pathlib import Path
import pandas as pd
import shutil
import DataMethods as dm

############################################################################################

# in the case of Savio, parent directory would be scratch/jellyname/recording name
parent_Dir = Path('/global/scratch/users/lilianzhang/Pink/20200707_Pink_218pm_cam2_1')
# parent_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/PinkTrainingData_Scratch')

# directory for video chunks within recording
videoDir = parent_Dir / 'Video_Chunks'

# home directory path for the recording
home_Dir = Path('/global/home/users/lilianzhang/Pink/20200707_Pink_218pm_cam2_1')
# home_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/PinkTrainingData_Home')

# Frame rate of recording
framerate = 120


def makeOutDir(outputDir, folderName):
    outdir = outputDir / folderName
    if not outdir.exists():
        outdir.mkdir()
    return outdir


stackDir = makeOutDir(parent_Dir, '{}_stacks'.format(parent_Dir.name))  # names it after recording_name_stacks
init_stackDir = makeOutDir(home_Dir, 'Initialization_Stack')
# sorts by alphabetical (should be correct in choosing first)
img_stack_dirs = sorted(stackDir / direc for direc in os.listdir(stackDir) if direc != '.DS_Store')
# create a list with all chunk videos in Video Directory as path objects
chunk_paths = sorted(videoDir / str(chunk) for chunk in os.listdir(videoDir) if chunk != '.DS_Store')
chunk_names = sorted(chunk.stem for chunk in chunk_paths)     # take stem of chunk_paths for video_names
image_stack_dir = sorted(stackDir for i in range(len(chunk_paths)))

print('directories made')

############################# DF Creation ######################################
num_frames_per_chunk = []
for chunk in img_stack_dirs:
    num_frames_per_chunk.append(len(dm.getFrameFilePaths(chunk)))
print('number frames in each chunk list is')
print(num_frames_per_chunk)

# num_frames_per_chunk = [chunk.iterdir() for chunk in chunk_paths]
pre_init_DF = pd.DataFrame({'RecordingName': parent_Dir.stem, 'RecordingDirPath': home_Dir, 'ChunkName': chunk_names,
                            'SavioChunkPath': img_stack_dirs, 'NumFramesInChunk': num_frames_per_chunk})

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
