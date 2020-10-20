import multiprocessing as mp
import VideoProcessingMethods as vpm
import DataMethods as dm
from pathlib import Path
import concurrent.futures
import sys
import os

postInit_DF_path = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/Short_Behavioral_Recordings/Home/NinaSimone/Initialization_DF/NinaSimone_PostInitializationDF_After.csv')

# params_df = dm.readCSV2pandasDF(postInit_DF_path)
#
# numChunks = len(params_df)
#
# print(params_df)
#
# for i in range(numChunks):
#     params_df = dm.readCSV2pandasDF(postInit_DF_path)
#     params_chunkRow = params_df.iloc[i]
#     print(params_chunkRow['ChunkName'])


# if __name__ == '__main__':
#     with mp.Pool(24) as p:
#         p.starmap(vpm.runFullVideoAnalysis, zip(range(numChunks), [postInit_DF_path]*numChunks))

# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         executor.map(vpm.runFullVideoAnalysis, zip(range(numChunks), [postInit_DF_path]*numChunks))

if __name__ == "__main__":
    arg = int(sys.argv[1])
    os.system('sleep {}'.format(arg))
    vpm.runFullVideoAnalysis(arg-1, postInit_DF_path)


# parallel python VideoProcessor_Main.py ::: $(seq 18)