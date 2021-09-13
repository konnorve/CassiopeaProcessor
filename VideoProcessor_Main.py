import multiprocessing as mp
import VideoProcessingMethods as vpm
import DataMethods as dm
from pathlib import Path
import concurrent.futures
import sys
from time import sleep


tmp_ImgStacks = Path(sys.argv[2]) # path to the /tmp/Image_Stacks directory within each overlay
postInit_DF_path = Path(sys.argv[1]) # path to csv

if __name__ == "__main__":
    
    postInit_DF = pd.read_csv(postInit_DF_path)
    
    for file in tmp_ImgStacks.iterdir():
        
        if file.is_dir() and not (postInit_DF[postInit_DF['ChunkName'] == file.stem].empty):
            
            chunkName_idx = postInit_DF[postInit_DF['ChunkName'] == file.stem].index[0]

            # arg = int(sys.argv[2]) # need to obtain arg as the index
            arg = chunkName_idx
            print(file.stem)
            print(arg)
            sleep(arg)
            print(postInit_DF_path)
            
            vpm.runFullVideoAnalysis(arg, postInit_DF_path)



# example running video processor in parallel on each chunk
# parallel python3 VideoProcessor_Main.py ::: $POSTINIT_DF_PATH ::: $(seq 60)
