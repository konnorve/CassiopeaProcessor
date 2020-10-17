import multiprocessing as mp
import VideoProcessingMethods as vpm
import DataMethods as dm
from pathlib import Path
import concurrent.futures
import sys

postInit_DF_path = Path('/Users/kve/Desktop/Clubs/Harland_Lab/Round_10/PinkTrainingData_Home/Initialization_DF/PinkTrainingData_Home_PostInitializationDF.csv')

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
    vpm.runFullVideoAnalysis(arg, postInit_DF_path)


# parallel python scratch_15.py ::: $(seq 15)