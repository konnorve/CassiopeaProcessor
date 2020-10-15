import multiprocessing as mp
import os
from pathlib import Path
import pandas as pd
import shutil
import DataMethods as dm
import concurrent.futures

############################################################################

# in the case of Savio, parent directory would be scratch/jellyname/recording name
parent_Dir = Path('/global/scratch/users/lilianzhang/Pink/20200707_Pink_218pm_cam2_1')
#parent_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/PinkTrainingData_Scratch')

# directory for video chunks within recording
videoDir = parent_Dir / 'Video_Chunks'

# home directory path for the recording
home_Dir = Path('/global/home/groups/fc_xenopus/Pink/20200707_Pink_218pm_cam2_1')
#home_Dir = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/PinkTrainingData_Home')

#Frame rate of recording
framerate = 120


def makeOutDir(outputDir, folderName):
    outdir = outputDir / folderName
    if not outdir.exists():
        outdir.mkdir()
    return outdir


def runFFMPEG(videoInPath, stackOutDir, frame_rate=framerate): # make frame rate variable
    try:
        os.system('ffmpeg -i {} -r {} -q 0 {}/%14d.jpg'.format(str(videoInPath), str(frame_rate), str(stackOutDir)))
        return True
    except:
        return False

stackDir = makeOutDir(parent_Dir, '{}_stacks'.format(parent_Dir.name))  # names it after recording_name_stacks

init_stackDir = makeOutDir(home_Dir, 'Initialization_Stack')
print(init_stackDir)

# if DEBUG: print("parent dir: {}".format(parentDir))
# if DEBUG: print("video dir: {}".format(videoDir))
# if DEBUG: print("stack dir: {}".format(stackDir))

# create a list with all chunk videos in Video Directory as path objects
chunk_paths = sorted(videoDir / str(chunk) for chunk in os.listdir(videoDir) if chunk != '.DS_Store')
chunk_names = sorted(chunk.stem for chunk in chunk_paths)     # take stem of chunk_paths for video_names
image_stack_dir = sorted(stackDir for i in range(len(chunk_paths)))

############################################################################
# create a directory for each chunk in image stacks
if __name__ == '__main__':
    with mp.Pool(24) as p:
        p.starmap(makeOutDir, zip(image_stack_dir, chunk_names))

print('directories made')

# sorts by alphabetical (should be correct in choosing first)
img_stack_dirs = sorted(stackDir / direc for direc in os.listdir(stackDir) if direc != '.DS_Store')

print('ffmpeg start')

# run ffmpeg on each and save to scratch directory
# if __name__ == '__main__':
#     with mp.Pool(24) as p:
#         p.starmap(runFFMPEG, zip(chunk_paths, img_stack_dirs))

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(runFFMPEG, zip(chunk_paths, img_stack_dirs))

print('ffmpeg complete')

####################### DF Creation #########################################

# chunk_paths and chunk_names should be in the same order
num_frames_per_chunk = []
for chunk in img_stack_dirs:
    num_frames_per_chunk.append(len(dm.getFrameFilePaths(chunk)))
print('Number frames in each chunk')
print(num_frames_per_chunk)

# num_frames_per_chunk = [chunk.iterdir() for chunk in chunk_paths]
pre_init_DF = pd.DataFrame({'RecordingName': parent_Dir.stem,
                            'RecordingDirPath': home_Dir,
                            'ChunkName': chunk_names,
                            'SavioChunkPath': img_stack_dirs,
                            'NumFramesInChunk': num_frames_per_chunk})

pre_init_DF['FrameRate'] = framerate

print('Pre-Init DF')
print(pre_init_DF)

# saves pre-initialization dataframe to home directory
init_Df_Dir = makeOutDir(home_Dir, 'Initialization_DF')
pre_init_DF.to_csv(init_Df_Dir / '{}_PreInitializationDF.csv'.format(home_Dir.stem))

print('DF Created')

#################### copies 30s image stack ##################################

imgs_for_init = dm.getFrameFilePaths(img_stack_dirs[0])[:3600]

print(imgs_for_init[:10])

for img in imgs_for_init:
    print(shutil.copy(str(img), str(init_stackDir)))

print('30s Stack Done')
############################################################################
# save initialization stack of first 30 images from ffmpeg to init_StackDir in home directory
# not sure if this will work on Savio but worked on local comp
# for image in sorted(os.listdir(image_stack_dir[0]))[:3600]:
#     if image != '.DS_Store':
#         img = stackDir / image_stack_dir[0] / image
#         print(img)
#         shutil.copy(str(img), str(init_stackDir))

# creates pre-initialization DF
# not sure about directory for recording dir path (home_Dir or videoDir?)
# chunk_paths and chunk_names should be in the same order
#num_frames_per_chunk = []
#for chunk in img_stack_dirs:                            # 'RecordingDirPath': home_Dir,
                            # 'ChunkName': chunk_names,
                            # 'SavioChunkPath': img_stack_dirs,
                            # 'NumFramesInChunk': num_frames_per_chunk
                            # })
#    num_frames_per_chunk.append(len(dm.getFrameFilePaths(chunk)))
#print(num_frames_per_chunk)

# num_frames_per_chunk = [chunk.iterdir() for chunk in chunk_paths]
#pre_init_DF = pd.DataFrame({'RecordingName': parent_Dir.stem,


#pre_init_DF['FrameRate'] = framerate

# saves pre-initialization dataframe to home directory
#init_Df_Dir = makeOutDir(home_Dir, 'Initialization_DF')
#pre_init_DF.to_csv(init_Df_Dir / '{}_PreInitializationDF.csv'.format(home_Dir.stem))

